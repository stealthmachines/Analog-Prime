/*
 * chat_win.c — Windows TUI chat client for Flash-MoE inference server
 *
 * Windows port of chat.m (macOS/POSIX Objective-C original).
 *
 *   — Winsock2 TCP HTTP/SSE streaming
 *   — ANSI markdown rendering (bold/italic/code/headers/bullets)
 *   — Session persistence in %APPDATA%\.flash-moe\sessions\*.jsonl
 *   — In-memory + file history with ReadConsoleW (basic line editing)
 *   — Tool call execution via cmd.exe (_popen)
 *
 * Build (clang):
 *   clang -O2 -D_CRT_SECURE_NO_WARNINGS -D_WIN32_WINNT=0x0601 ^
 *         chat_win.c -o chat_win.exe -lws2_32
 *
 * Build (MSVC):
 *   cl /O2 /TC /D_CRT_SECURE_NO_WARNINGS chat_win.c ws2_32.lib /Fe:chat_win.exe
 *
 * Run:
 *   chat_win.exe [--port 8000] [--max-tokens N] [--show-think]
 *                [--resume <id>] [--sessions] [--hdgl] [--help]
 *
 * Licensed per https://zchg.org/t/legal-notice-copyright-applicable-ip-and-licensing-read-me/440
 */

#ifndef _WIN32_WINNT
#  define _WIN32_WINNT 0x0601   /* Windows 7+ */
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#  define _CRT_SECURE_NO_WARNINGS
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <direct.h>     /* _mkdir */
#include <io.h>         /* _findfirst, _findnext, _findclose, _finddata_t */
#include <sys/stat.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <ctype.h>

#pragma comment(lib, "ws2_32.lib")

/* ── Constants ─────────────────────────────────────────────────────────── */
#define MAX_INPUT_LINE  4096
#define MAX_RESPONSE    (1 * 1024 * 1024)
#define HISTORY_MAX     500
#define RECV_BUF_SIZE   65536
#define LINE_BUF_SIZE   65536

/* ── Timing ────────────────────────────────────────────────────────────── */
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}

/* ── ANSI + UTF-8 console init ─────────────────────────────────────────── */
static void enable_ansi(void) {
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(hout, &mode)) {
        SetConsoleMode(hout, mode
            | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            | ENABLE_PROCESSED_OUTPUT);
    }
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}

/* ── JSON helpers ──────────────────────────────────────────────────────── */
static int json_escape(const char *src, char *buf, int bufsize) {
    int j = 0;
    for (int i = 0; src[i] && j < bufsize - 6; i++) {
        switch (src[i]) {
            case '"':  buf[j++]='\\'; buf[j++]='"';  break;
            case '\\': buf[j++]='\\'; buf[j++]='\\'; break;
            case '\n': buf[j++]='\\'; buf[j++]='n';  break;
            case '\r': buf[j++]='\\'; buf[j++]='r';  break;
            case '\t': buf[j++]='\\'; buf[j++]='t';  break;
            default:   buf[j++]=src[i]; break;
        }
    }
    buf[j] = 0;
    return j;
}

/* Decode JSON string escapes in-place, returns new length */
static int json_unescape(const char *src, char *dst, int maxlen) {
    int di = 0;
    for (int i = 0; src[i] && src[i] != '"' && di < maxlen - 1; i++) {
        if (src[i] == '\\' && src[i+1]) {
            i++;
            switch (src[i]) {
                case 'n':  dst[di++] = '\n'; break;
                case 't':  dst[di++] = '\t'; break;
                case '"':  dst[di++] = '"';  break;
                case '\\': dst[di++] = '\\'; break;
                case 'r':  dst[di++] = '\r'; break;
                default:   dst[di++] = src[i]; break;
            }
        } else {
            dst[di++] = src[i];
        }
    }
    dst[di] = 0;
    return di;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Session persistence
 * ══════════════════════════════════════════════════════════════════════════ */

static char g_sessions_dir[MAX_PATH];
static char g_history_file[MAX_PATH];

static void init_sessions_dir(void) {
    const char *appdata = getenv("APPDATA");
    if (!appdata) appdata = getenv("USERPROFILE");
    if (!appdata) appdata = ".";

    char base[MAX_PATH];
    _snprintf(base, sizeof(base), "%s\\.flash-moe", appdata);
    _mkdir(base);

    _snprintf(g_sessions_dir, sizeof(g_sessions_dir), "%s\\sessions", base);
    _mkdir(g_sessions_dir);

    _snprintf(g_history_file, sizeof(g_history_file), "%s\\history", base);
}

static void session_path(const char *session_id, char *path, size_t pathsize) {
    _snprintf(path, pathsize, "%s\\%s.jsonl", g_sessions_dir, session_id);
}

static void session_save_turn(const char *session_id, const char *role, const char *content) {
    char path[MAX_PATH];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "a");
    if (!f) return;
    char *escaped = (char *)malloc(MAX_RESPONSE * 2);
    if (escaped) {
        json_escape(content, escaped, MAX_RESPONSE * 2);
        fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
        free(escaped);
    }
    fclose(f);
}

static int session_load(const char *session_id) {
    char path[MAX_PATH];
    session_path(session_id, path, sizeof(path));
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    printf("[resuming session %s]\n\n", session_id);
    int turns = 0;
    char *line = (char *)malloc(MAX_RESPONSE);
    if (!line) { fclose(f); return 0; }

    while (fgets(line, MAX_RESPONSE, f)) {
        char *role_start = strstr(line, "\"role\":\"");
        char *content_start = strstr(line, "\"content\":\"");
        if (!role_start || !content_start) continue;

        role_start += 8;
        char role[32]; int ri = 0;
        while (*role_start && *role_start != '"' && ri < 31) role[ri++] = *role_start++;
        role[ri] = 0;

        content_start += 11;
        char *content = (char *)malloc(MAX_RESPONSE);
        if (!content) continue;
        json_unescape(content_start, content, MAX_RESPONSE);

        if (strcmp(role, "user") == 0) {
            printf("\033[1m> %s\033[0m\n\n", content);
        } else if (strcmp(role, "assistant") == 0) {
            printf("%s\n\n", content);
        }
        free(content);
        turns++;
    }
    free(line);
    fclose(f);
    if (turns > 0) printf("[%d turns loaded]\n\n", turns);
    return turns;
}

static void session_list(void) {
    char pattern[MAX_PATH];
    _snprintf(pattern, sizeof(pattern), "%s\\*.jsonl", g_sessions_dir);

    struct _finddata_t fd;
    intptr_t handle = _findfirst(pattern, &fd);
    if (handle == -1) {
        printf("No sessions found.\n\n");
        return;
    }

    printf("Recent sessions:\n");
    int count = 0;
    do {
        /* strip .jsonl extension for display */
        char name[MAX_PATH];
        strncpy(name, fd.name, sizeof(name) - 1);
        name[sizeof(name) - 1] = 0;
        char *dot = strrchr(name, '.');
        if (!dot) continue;
        *dot = 0;

        char path[MAX_PATH];
        _snprintf(path, sizeof(path), "%s\\%s.jsonl", g_sessions_dir, name);

        /* Count turns (lines) */
        FILE *f = fopen(path, "r");
        int lines = 0;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), f)) lines++;
            fclose(f);
        }
        printf("  %-40s  (%d turns)\n", name, lines);
        count++;
    } while (_findnext(handle, &fd) == 0);

    _findclose(handle);
    if (count == 0) printf("  (none)\n");
    printf("\n");
}

/* ══════════════════════════════════════════════════════════════════════════
 * History — in-memory ring buffer + file save/load
 * ══════════════════════════════════════════════════════════════════════════ */

static char *g_history[HISTORY_MAX];
static int   g_history_count = 0;

static void history_add(const char *line) {
    if (!line || !*line) return;
    /* Deduplicate: skip if same as last */
    if (g_history_count > 0 && strcmp(g_history[g_history_count - 1], line) == 0) return;
    if (g_history_count < HISTORY_MAX) {
        g_history[g_history_count++] = _strdup(line);
    } else {
        free(g_history[0]);
        memmove(g_history, g_history + 1, (HISTORY_MAX - 1) * sizeof(char *));
        g_history[HISTORY_MAX - 1] = _strdup(line);
    }
}

static void history_save(void) {
    FILE *f = fopen(g_history_file, "w");
    if (!f) return;
    int start = g_history_count > 200 ? g_history_count - 200 : 0;
    for (int i = start; i < g_history_count; i++)
        fprintf(f, "%s\n", g_history[i]);
    fclose(f);
}

static void history_load(void) {
    FILE *f = fopen(g_history_file, "r");
    if (!f) return;
    char buf[MAX_INPUT_LINE];
    while (fgets(buf, sizeof(buf), f)) {
        int len = (int)strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = 0;
        if (len > 0) history_add(buf);
    }
    fclose(f);
}

/* ── readline_win: ReadConsoleW-based prompt with history navigation ───── */
/*
 * Uses SetConsoleMode to enter raw character-at-a-time mode so we can
 * intercept Up/Down arrows for history navigation while still supporting
 * normal editing (backspace, left/right via Windows built-in if we use
 * the VT input mode, or manually if raw).
 *
 * Implementation: raw INPUT_RECORD loop with VK_ codes.
 */
static char *readline_win(const char *prompt) {
    static int  hist_pos = -1;   /* -1 = new input */
    static char saved_line[MAX_INPUT_LINE] = {0};

    HANDLE hin  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);

    DWORD old_mode;
    GetConsoleMode(hin, &old_mode);
    /* Raw: disable line input and echo; enable processed input for Ctrl+C */
    SetConsoleMode(hin, ENABLE_PROCESSED_INPUT | ENABLE_VIRTUAL_TERMINAL_INPUT);

    /* Print prompt */
    DWORD written;
    {
        WCHAR wprompt[256];
        int wlen = MultiByteToWideChar(CP_UTF8, 0, prompt, -1, wprompt, 255);
        WriteConsoleW(hout, wprompt, wlen > 0 ? wlen - 1 : 0, &written, NULL);
    }

    char  buf[MAX_INPUT_LINE];
    int   len   = 0;
    int   cur   = 0;    /* cursor position in buf */
    hist_pos = -1;      /* reset history nav */
    buf[0] = 0;

    for (;;) {
        INPUT_RECORD ir;
        DWORD nread;
        if (!ReadConsoleInputW(hin, &ir, 1, &nread)) break;
        if (nread == 0) continue;
        if (ir.EventType != KEY_EVENT || !ir.Event.KeyEvent.bKeyDown) continue;

        WORD vk    = ir.Event.KeyEvent.wVirtualKeyCode;
        WCHAR wch  = ir.Event.KeyEvent.uChar.UnicodeChar;
        DWORD ctrl = ir.Event.KeyEvent.dwControlKeyState;
        (void)ctrl;

        if (vk == VK_RETURN || wch == L'\r') {
            /* Newline — also catches ConPTY where VK=0, uChar='\r' */
            WriteConsoleW(hout, L"\r\n", 2, &written, NULL);
            break;
        }

        if (vk == VK_BACK) {
            if (cur > 0) {
                /* remove char before cursor */
                memmove(buf + cur - 1, buf + cur, len - cur + 1);
                cur--; len--;
                /* redraw from cursor */
                printf("\r\033[2K");
                printf("%s%s", prompt, buf);
                /* reposition cursor */
                if (cur < len) {
                    int back = len - cur;
                    printf("\033[%dD", back);
                }
                fflush(stdout);
            }
            continue;
        }

        if (vk == VK_DELETE) {
            if (cur < len) {
                memmove(buf + cur, buf + cur + 1, len - cur);
                len--;
                printf("\r\033[2K%s%s", prompt, buf);
                if (cur < len) printf("\033[%dD", len - cur);
                fflush(stdout);
            }
            continue;
        }

        if (vk == VK_LEFT) {
            if (cur > 0) { cur--; printf("\033[1D"); fflush(stdout); }
            continue;
        }
        if (vk == VK_RIGHT) {
            if (cur < len) { cur++; printf("\033[1C"); fflush(stdout); }
            continue;
        }
        if (vk == VK_HOME) {
            if (cur > 0) { printf("\033[%dD", cur); fflush(stdout); cur = 0; }
            continue;
        }
        if (vk == VK_END) {
            if (cur < len) { printf("\033[%dC", len - cur); fflush(stdout); cur = len; }
            continue;
        }

        if (vk == VK_UP) {
            /* Navigate history backwards */
            if (g_history_count == 0) continue;
            if (hist_pos == -1) {
                /* Save current input */
                strncpy(saved_line, buf, MAX_INPUT_LINE - 1);
                saved_line[MAX_INPUT_LINE - 1] = 0;
                hist_pos = g_history_count - 1;
            } else if (hist_pos > 0) {
                hist_pos--;
            }
            strncpy(buf, g_history[hist_pos], MAX_INPUT_LINE - 1);
            buf[MAX_INPUT_LINE - 1] = 0;
            len = (int)strlen(buf);
            cur = len;
            printf("\r\033[2K%s%s", prompt, buf);
            fflush(stdout);
            continue;
        }

        if (vk == VK_DOWN) {
            if (hist_pos == -1) continue;
            if (hist_pos < g_history_count - 1) {
                hist_pos++;
                strncpy(buf, g_history[hist_pos], MAX_INPUT_LINE - 1);
            } else {
                hist_pos = -1;
                strncpy(buf, saved_line, MAX_INPUT_LINE - 1);
            }
            buf[MAX_INPUT_LINE - 1] = 0;
            len = (int)strlen(buf);
            cur = len;
            printf("\r\033[2K%s%s", prompt, buf);
            fflush(stdout);
            continue;
        }

        /* Ctrl+C — return NULL */
        if (vk == 'C' && (ctrl & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED))) {
            WriteConsoleW(hout, L"^C\r\n", 4, &written, NULL);
            SetConsoleMode(hin, old_mode);
            return NULL;
        }

        /* Ctrl+D on empty line — EOF */
        if (vk == 'D' && (ctrl & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) && len == 0) {
            WriteConsoleW(hout, L"\r\n", 2, &written, NULL);
            SetConsoleMode(hin, old_mode);
            return NULL;
        }

        /* Printable Unicode character */
        if (wch >= 0x20 || wch == '\t') {
            /* Convert to UTF-8 */
            char mb[8] = {0};
            int mb_len = WideCharToMultiByte(CP_UTF8, 0, &wch, 1, mb, sizeof(mb)-1, NULL, NULL);
            if (mb_len > 0 && len + mb_len < MAX_INPUT_LINE - 1) {
                memmove(buf + cur + mb_len, buf + cur, len - cur + 1);
                memcpy(buf + cur, mb, mb_len);
                len += mb_len;
                cur += mb_len;
                /* Print the updated suffix */
                if (cur == len) {
                    /* Cursor at end: just print new chars */
                    fwrite(mb, 1, mb_len, stdout);
                } else {
                    /* Cursor mid-line: redraw from cursor */
                    printf("\033[s");          /* save cursor */
                    fwrite(buf + cur - mb_len, 1, len - (cur - mb_len), stdout);
                    printf("\033[u");          /* restore cursor */
                    printf("\033[%dC", mb_len); /* advance by mb_len */
                    /* Simpler: just full redraw */
                    printf("\r\033[2K%s%s", prompt, buf);
                    if (cur < len) printf("\033[%dD", len - cur);
                }
                fflush(stdout);
            }
        }
    }

    SetConsoleMode(hin, old_mode);

    if (len == 0) {
        char *empty = (char *)malloc(1);
        empty[0] = 0;
        return empty;
    }

    char *result = _strdup(buf);
    history_add(result);
    return result;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Session ID generation
 * ══════════════════════════════════════════════════════════════════════════ */
static void generate_session_id(char *buf, size_t bufsize) {
    DWORD pid = GetCurrentProcessId();
    ULONGLONG tick = GetTickCount64();
    _snprintf(buf, bufsize, "chat-%lu-%llu",
              (unsigned long)pid, (unsigned long long)tick);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Winsock2 TCP helpers
 * ══════════════════════════════════════════════════════════════════════════ */

/* Buffered recv-based line reader (replaces fdopen on POSIX) */
typedef struct {
    SOCKET sock;
    char   buf[RECV_BUF_SIZE];
    int    buf_len;
    int    buf_pos;
    int    eof;
} RecvBuf;

static void recvbuf_init(RecvBuf *rb, SOCKET sock) {
    rb->sock    = sock;
    rb->buf_len = 0;
    rb->buf_pos = 0;
    rb->eof     = 0;
}

/* Read one line (LF-terminated) into line[0..maxlen-1], strip CR.
 * Returns 1 on success, 0 on EOF/error with no bytes read.
 * If EOF arrived mid-line, returns 1 with whatever was buffered. */
static int recvbuf_getline(RecvBuf *rb, char *line, int maxlen) {
    int llen = 0;
    for (;;) {
        if (rb->buf_pos >= rb->buf_len) {
            if (rb->eof) {
                line[llen] = 0;
                return llen > 0 ? 1 : 0;
            }
            int n = recv(rb->sock, rb->buf, RECV_BUF_SIZE, 0);
            if (n <= 0) {
                rb->eof = 1;
                line[llen] = 0;
                return llen > 0 ? 1 : 0;
            }
            rb->buf_len = n;
            rb->buf_pos = 0;
        }
        char c = rb->buf[rb->buf_pos++];
        if (c == '\n') {
            if (llen > 0 && line[llen-1] == '\r') llen--;
            line[llen] = 0;
            return 1;
        }
        if (llen < maxlen - 1) line[llen++] = c;
    }
}

/* Open TCP connection to 127.0.0.1:port, send HTTP POST, return socket */
static SOCKET send_chat_request(int port, const char *user_message,
                                 int max_tokens, const char *session_id) {
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) { fprintf(stderr, "[error] socket()\n"); return INVALID_SOCKET; }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((u_short)port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
        fprintf(stderr, "\n[error] Cannot connect to server on port %d.\n", port);
        closesocket(sock);
        return INVALID_SOCKET;
    }

    char *escaped = (char *)malloc(MAX_INPUT_LINE * 2);
    if (!escaped) { closesocket(sock); return INVALID_SOCKET; }
    json_escape(user_message, escaped, MAX_INPUT_LINE * 2);

    char *body = (char *)malloc(MAX_INPUT_LINE * 3);
    char *request = (char *)malloc(MAX_INPUT_LINE * 4);
    if (!body || !request) {
        free(escaped); free(body); free(request);
        closesocket(sock); return INVALID_SOCKET;
    }

    int body_len = _snprintf(body, MAX_INPUT_LINE * 3,
        "{\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],"
        "\"max_tokens\":%d,\"stream\":true,\"session_id\":\"%s\"}",
        escaped, max_tokens, session_id);

    int req_len = _snprintf(request, MAX_INPUT_LINE * 4,
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:%d\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        port, body_len, body);

    send(sock, request, req_len, 0);
    free(escaped); free(body); free(request);
    return sock;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Streaming markdown renderer — stateful ANSI escape emitter
 * ══════════════════════════════════════════════════════════════════════════
 * Handles: **bold**, *italic*, `inline code`, ```blocks```, # headers,
 *          - bullet lists, 1. numbered lists.
 * Ported verbatim from chat.m (logic unchanged; ANSI constants same).
 */

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_ITALIC  "\033[3m"
#define ANSI_CODE    "\033[36m"
#define ANSI_CODEBLK "\033[48;5;236m\033[38;5;252m"
#define ANSI_CODEBLK_LINE "\033[48;5;236m\033[K"
#define ANSI_HEADER  "\033[1;34m"
#define ANSI_DIM     "\033[2m"

typedef struct {
    int bold;
    int italic;
    int code_inline;
    int code_block;
    int skip_lang;
    int line_start;
    char pending[8];
    int  pending_len;
} MdState;

static MdState g_md;

static void md_reset(void) {
    memset(&g_md, 0, sizeof(g_md));
    g_md.line_start = 1;
}

static void md_print(const char *text) {
    for (int i = 0; text[i]; i++) {
        char c = text[i];

        if (g_md.skip_lang) {
            if (c == '\n') {
                g_md.skip_lang = 0;
                printf(ANSI_CODEBLK ANSI_CODEBLK_LINE "\n");
            }
            continue;
        }

        /* Code block ``` */
        if (c == '`' && text[i+1] == '`' && text[i+2] == '`') {
            if (g_md.code_block) {
                printf(ANSI_RESET "\n");
                g_md.code_block = 0;
            } else {
                g_md.code_block = 1;
                g_md.skip_lang  = 1;
            }
            i += 2;
            continue;
        }

        if (g_md.code_block) {
            printf(ANSI_CODEBLK);
            if (c == '\n') printf(ANSI_CODEBLK_LINE "\n");
            else putchar(c);
            continue;
        }

        /* Inline code ` */
        if (c == '`') {
            if (g_md.code_inline) { printf(ANSI_RESET); g_md.code_inline = 0; }
            else                  { printf(ANSI_CODE);  g_md.code_inline = 1; }
            continue;
        }
        if (g_md.code_inline) { putchar(c); continue; }

        /* Headers */
        if (g_md.line_start && c == '#') {
            while (text[i] == '#') i++;
            while (text[i] == ' ') i++;
            printf(ANSI_HEADER);
            while (text[i] && text[i] != '\n') { putchar(text[i]); i++; }
            printf(ANSI_RESET);
            if (text[i] == '\n') { putchar('\n'); g_md.line_start = 1; }
            continue;
        }

        /* Bullet lists */
        if (g_md.line_start && (c == '-' || c == '*' || c == ' ')) {
            int indent = 0, peek = i;
            while (text[peek] == ' ' || text[peek] == '\t') { indent++; peek++; }
            char marker = text[peek];
            if (marker == '-' || marker == '*') {
                char after = text[peek + 1];
                if (marker == '-' && (after == ' ' || after == '\0')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m\xE2\x80\xA2\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--;
                    g_md.line_start = 0;
                    continue;
                }
                if (marker == '*' && after != '*' && (after == ' ' || after == '\0' || after == '\t')) {
                    int depth = indent / 2;
                    for (int d = 0; d < depth + 1; d++) printf("  ");
                    printf("\033[33m\xE2\x80\xA2\033[0m ");
                    i = peek + 1;
                    while (text[i] == ' ' || text[i] == '\t') i++;
                    i--;
                    g_md.line_start = 0;
                    continue;
                }
            }
        }

        /* Numbered lists */
        if (g_md.line_start && c >= '0' && c <= '9') {
            int num_start = i;
            while (text[i] >= '0' && text[i] <= '9') i++;
            if (text[i] == '.' && text[i+1] == ' ') {
                printf("  \033[33m");
                for (int j = num_start; j <= i; j++) putchar(text[j]);
                printf("\033[0m");
                i++;
                g_md.line_start = 0;
                continue;
            }
            i = num_start; c = text[i];
        }

        /* Bold ** */
        if (c == '*' && text[i+1] == '*') {
            if (g_md.bold) { printf(ANSI_RESET); g_md.bold = 0; }
            else           { printf(ANSI_BOLD);  g_md.bold = 1; }
            i++;
            continue;
        }

        /* Italic * */
        if (c == '*' && text[i+1] != '*') {
            if (g_md.italic) { printf(ANSI_RESET);   g_md.italic = 0; }
            else             { printf(ANSI_ITALIC); g_md.italic = 1; }
            continue;
        }

        if (c == '\n') g_md.line_start = 1;
        else           g_md.line_start = 0;
        putchar(c);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * SSE streaming response reader
 * ══════════════════════════════════════════════════════════════════════════ */
static char *stream_response(SOCKET sock, int show_thinking) {
    RecvBuf rb;
    recvbuf_init(&rb, sock);

    int header_done = 0, in_think = 0, tokens = 0;
    double t_start = now_ms(), t_first = 0;
    md_reset();

    char *response = (char *)calloc(1, MAX_RESPONSE);
    if (!response) { closesocket(sock); return NULL; }
    int resp_len = 0;

    char *line = (char *)malloc(LINE_BUF_SIZE);
    char *decoded = (char *)malloc(LINE_BUF_SIZE);
    if (!line || !decoded) {
        free(response); free(line); free(decoded);
        closesocket(sock); return NULL;
    }

    while (recvbuf_getline(&rb, line, LINE_BUF_SIZE)) {
        if (!header_done) {
            if (line[0] == '\0') header_done = 1;
            continue;
        }

        if (strncmp(line, "data: ", 6) != 0) continue;
        if (strncmp(line + 6, "[DONE]", 6) == 0) break;

        char *ck = strstr(line + 6, "\"content\":\"");
        if (!ck) continue;
        ck += 11;

        int di = json_unescape(ck, decoded, LINE_BUF_SIZE);
        if (!di) continue;

        if (strstr(decoded, "<think>"))   in_think = 1;
        if (strstr(decoded, "</think>")) { in_think = 0; tokens++; continue; }
        tokens++;
        if (!t_first) t_first = now_ms();

        if (!in_think && resp_len + di < MAX_RESPONSE - 1) {
            memcpy(response + resp_len, decoded, di);
            resp_len += di;
            response[resp_len] = 0;
        }

        if (in_think && !show_thinking) continue;
        if (in_think) printf(ANSI_DIM "%s" ANSI_RESET, decoded);
        else          md_print(decoded);
        fflush(stdout);
    }

    free(line); free(decoded);
    closesocket(sock);

    printf(ANSI_RESET);
    double gen_time  = t_first > 0 ? now_ms() - t_first : 0;
    int    gen_tokens = tokens > 1 ? tokens - 1 : 0;
    printf("\n\n");
    if (gen_tokens > 0 && gen_time > 0.0)
        printf("[%d tokens, %.1f tok/s]\n\n",
               tokens, gen_tokens * 1000.0 / gen_time);

    return response;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Tool call handler
 * ══════════════════════════════════════════════════════════════════════════ */
/* Parse a <tool_call>...</tool_call> block and extract the bash command.
 * Returns 1 if a command was found and written to `command`. */
static int parse_tool_call(const char *response, char *command, int cmdsize) {
    const char *tc_start = strstr(response, "<tool_call>");
    const char *tc_end   = strstr(response, "</tool_call>");
    if (!tc_start || !tc_end) return 0;

    tc_start += 11;
    char tc_body[4096] = {0};
    int tc_len = (int)(tc_end - tc_start);
    if (tc_len > (int)sizeof(tc_body) - 1) tc_len = (int)sizeof(tc_body) - 1;
    memcpy(tc_body, tc_start, tc_len);

    int ci = 0;
    /* JSON: "command":"..." */
    char *cmd_key = strstr(tc_body, "\"command\"");
    if (cmd_key) {
        cmd_key = strchr(cmd_key + 9, '"');
        if (cmd_key) {
            cmd_key++;
            json_unescape(cmd_key, command, cmdsize);
            ci = (int)strlen(command);
        }
    }
    /* <arg_value>...</arg_value> */
    if (ci == 0) {
        char *av = strstr(tc_body, "<arg_value>");
        if (av) {
            av += 11;
            char *av_end = strstr(av, "</arg_value>");
            if (!av_end) av_end = strstr(av, "<");
            if (av_end) {
                int avlen = (int)(av_end - av);
                if (avlen > cmdsize - 1) avlen = cmdsize - 1;
                memcpy(command, av, avlen);
                ci = avlen;
                while (ci > 0 && (command[ci-1] == '\n' || command[ci-1] == ' ')) ci--;
                command[ci] = 0;
            }
        }
    }
    return ci > 0 ? 1 : 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * main
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    /* ── Winsock init ─────────────────────────────────────────────────── */
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "[error] WSAStartup failed.\n");
        return 1;
    }

    enable_ansi();

    /* ── Arg parsing ──────────────────────────────────────────────────── */
    int         port          = 8000;
    int         max_tokens    = 8192;
    int         show_thinking = 0;
    const char *resume_id     = NULL;
    int         list_sessions = 0;
    int         hdgl_mode     = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--show-think") == 0)
            show_thinking = 1;
        else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc)
            resume_id = argv[++i];
        else if (strcmp(argv[i], "--sessions") == 0)
            list_sessions = 1;
        else if (strcmp(argv[i], "--hdgl") == 0)
            hdgl_mode = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: chat_win.exe [options]\n");
            printf("  --port N         Server port (default: 8000)\n");
            printf("  --max-tokens N   Max response tokens (default: 8192)\n");
            printf("  --show-think     Show <think> blocks (dimmed)\n");
            printf("  --resume ID      Resume a previous session\n");
            printf("  --sessions       List saved sessions\n");
            printf("  --hdgl           HDGL-28 APA lattice mode banner\n");
            printf("  --help           This message\n");
            WSACleanup();
            return 0;
        }
    }

    init_sessions_dir();
    history_load();

    if (list_sessions) {
        session_list();
        WSACleanup();
        return 0;
    }

    /* ── Session setup ────────────────────────────────────────────────── */
    char session_id[64];
    if (resume_id) {
        strncpy(session_id, resume_id, sizeof(session_id) - 1);
        session_id[sizeof(session_id) - 1] = 0;
    } else {
        generate_session_id(session_id, sizeof(session_id));
    }

    /* ── Banner ───────────────────────────────────────────────────────── */
    printf("\033[1;36m");
    printf("==================================================\n");
    printf("  Qwen3.5-397B-A17B Chat  (Flash-MoE / Windows)\n");
    printf("==================================================\033[0m\n");
    printf("  Server:  \033[32mhttp://localhost:%d\033[0m\n", port);
    printf("  Session: \033[33m%s\033[0m%s\n",
           session_id, resume_id ? " \033[2m(resumed)\033[0m" : "");
    printf("\n  Commands: /quit  /exit  /clear  /sessions  /history\n");
    printf("\033[1;36m==================================================\033[0m\n\n");

    if (hdgl_mode) {
        printf("\n\033[1;35m[HDGL-28] Hypervisor-MoE Bolstered"
               " - BootloaderZ APA Lattice + Prismatic Router\033[0m\n");
        printf("\033[35mYou, Hypervisor; and Me\033[0m\n\n");
    }

    /* ── Health check ─────────────────────────────────────────────────── */
    {
        SOCKET check = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port   = htons((u_short)port);
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(check, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
            fprintf(stderr, "\033[31mServer not running on port %d.\033[0m\n", port);
            fprintf(stderr, "Start with: nonmetal_infer.exe --serve %d\n"
                            "       or:  infer --serve %d\n\n", port, port);
            closesocket(check);
            WSACleanup();
            return 1;
        }
        closesocket(check);
    }

    /* ── Resume ───────────────────────────────────────────────────────── */
    if (resume_id) {
        int turns = session_load(session_id);
        if (turns == 0)
            printf("\033[33mNo session found with ID: %s\033[0m\n\n", session_id);
    }

    printf("\033[32mReady to chat.\033[0m\n\n");

    /* ── Main chat loop ───────────────────────────────────────────────── */
    for (;;) {
        char *line = readline_win("\033[1m> \033[0m");
        if (!line) {
            printf("\nGoodbye.\n");
            break;
        }

        size_t input_len = strlen(line);
        if (input_len == 0) { free(line); continue; }

        /* Save to history file */
        history_save();

        char input_line[MAX_INPUT_LINE];
        strncpy(input_line, line, MAX_INPUT_LINE - 1);
        input_line[MAX_INPUT_LINE - 1] = 0;
        free(line);

        /* Built-in commands */
        if (strcmp(input_line, "/quit") == 0 || strcmp(input_line, "/exit") == 0) {
            printf("Goodbye.\n");
            break;
        }
        if (strcmp(input_line, "/clear") == 0) {
            generate_session_id(session_id, sizeof(session_id));
            printf("\033[2m[new session: %s]\033[0m\n\n", session_id);
            continue;
        }
        if (strcmp(input_line, "/sessions") == 0) {
            session_list();
            continue;
        }
        if (strcmp(input_line, "/history") == 0) {
            int start = g_history_count > 20 ? g_history_count - 20 : 0;
            printf("\033[2mRecent history:\033[0m\n");
            for (int i = start; i < g_history_count; i++)
                printf("  \033[2m%3d\033[0m  %s\n", i - start + 1, g_history[i]);
            printf("\n");
            continue;
        }

        /* Send to server */
        session_save_turn(session_id, "user", input_line);

        SOCKET sock = send_chat_request(port, input_line, max_tokens, session_id);
        if (sock == INVALID_SOCKET) continue;

        printf("\n");
        char *response = stream_response(sock, show_thinking);

        if (response && strlen(response) > 0)
            session_save_turn(session_id, "assistant", response);

        /* Tool call loop */
        while (response && strstr(response, "<tool_call>")) {
            char command[4096] = {0};
            if (!parse_tool_call(response, command, sizeof(command))) break;

            printf("\033[33m$ %s\033[0m\n", command);
            printf("\033[2m[execute? y/n] \033[0m");
            fflush(stdout);

            int ch = getchar();
            while (getchar() != '\n');
            if (ch != 'y' && ch != 'Y') {
                printf("\033[2m[skipped]\033[0m\n");
                free(response);
                response = NULL;
                break;
            }

            /* Execute via cmd.exe */
            FILE *proc = _popen(command, "r");
            char *output = (char *)calloc(1, 65536);
            int out_len = 0;
            if (proc && output) {
                int oc;
                while (out_len < 65534 && (oc = fgetc(proc)) != EOF)
                    output[out_len++] = (char)oc;
                output[out_len] = 0;
                _pclose(proc);
            }

            if (out_len > 0) {
                printf("\033[2m%s\033[0m", output);
                if (output[out_len-1] != '\n') printf("\n");
            }

            char *tool_msg = (char *)malloc(out_len + 256);
            if (tool_msg) {
                _snprintf(tool_msg, out_len + 256,
                          "<tool_response>\n%s</tool_response>", output ? output : "");
            }
            free(output);
            free(response);

            sock = send_chat_request(port, tool_msg ? tool_msg : "", max_tokens, session_id);
            free(tool_msg);
            if (sock == INVALID_SOCKET) { response = NULL; break; }

            printf("\n");
            response = stream_response(sock, show_thinking);
            if (response && strlen(response) > 0)
                session_save_turn(session_id, "assistant", response);
        }

        free(response);
    }

    WSACleanup();
    return 0;
}
