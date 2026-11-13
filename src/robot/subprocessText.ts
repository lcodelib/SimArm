/** 子进程 / CLI 文本解码（兼容浏览器与 Node） */

function hasCjk(s: string): boolean {
  return /[\u4e00-\u9fff]/.test(s);
}

/** UTF-8 被按 Latin-1 误读时的常见乱码修复 */
function latin1MisreadUtf8Fix(s: string): string {
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i) & 0xff;
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

/** 将子进程 stdout/stderr 字节解码为字符串 */
export function decodeSubprocessOutput(buf: Uint8Array): string {
  if (buf.length === 0) return "";

  const utf8 = new TextDecoder("utf-8", { fatal: false }).decode(buf);
  if (hasCjk(utf8) && !utf8.includes("\uFFFD")) return utf8;

  try {
    const gbk = new TextDecoder("gbk").decode(buf);
    if (hasCjk(gbk)) return gbk;
  } catch {
    /* 部分环境无 gbk TextDecoder */
  }

  const latin1Fix = latin1MisreadUtf8Fix(utf8);
  if (hasCjk(latin1Fix)) return latin1Fix;

  return utf8;
}

/** 修复已被错误解码的中文错误信息 */
export function normalizeCliErrorMessage(msg: string): string {
  if (!msg || hasCjk(msg)) return msg;
  const fixed = latin1MisreadUtf8Fix(msg);
  return hasCjk(fixed) ? fixed : msg;
}
