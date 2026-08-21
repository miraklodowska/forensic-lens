/**
 * Size + SHA-256 validation for downloaded model artifacts.
 *
 * Both checks are mandatory and neither is ever relaxed: a byte-length mismatch
 * fails before hashing (cheap rejection of truncated or redirected downloads),
 * and the digest must match the manifest exactly.
 */

export interface ArtifactExpectation {
  readonly sizeBytes: number;
  readonly sha256: string;
}

export type IntegrityCheck = 'size' | 'sha256';

/**
 * Only `digest` is ever used. Narrowing the dependency to this shape lets the
 * same code accept the browser's `crypto.subtle`, Node's `webcrypto.subtle`,
 * and a counting test double without casts.
 */
export interface DigestProvider {
  digest(algorithm: 'SHA-256', data: BufferSource): Promise<ArrayBuffer>;
}

export class IntegrityError extends Error {
  readonly check: IntegrityCheck;

  constructor(check: IntegrityCheck, message: string) {
    super(message);
    this.name = 'IntegrityError';
    this.check = check;
  }
}

function toHex(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let hex = '';
  for (const byte of bytes) hex += byte.toString(16).padStart(2, '0');
  return hex;
}

export async function sha256Hex(
  bytes: Uint8Array,
  subtle: DigestProvider = globalThis.crypto.subtle,
): Promise<string> {
  // Copy into a freshly-owned buffer so a Uint8Array view over a larger
  // ArrayBuffer never hashes its neighbours.
  const view = new Uint8Array(bytes.byteLength);
  view.set(bytes);
  return toHex(await subtle.digest('SHA-256', view));
}

export async function verifyArtifact(
  bytes: Uint8Array,
  expected: ArtifactExpectation,
  subtle: DigestProvider = globalThis.crypto.subtle,
): Promise<Uint8Array> {
  if (bytes.byteLength !== expected.sizeBytes) {
    throw new IntegrityError(
      'size',
      `model size mismatch: expected ${expected.sizeBytes} bytes, received ${bytes.byteLength} bytes`,
    );
  }
  const actual = await sha256Hex(bytes, subtle);
  if (actual !== expected.sha256) {
    throw new IntegrityError(
      'sha256',
      `model sha256 mismatch: expected ${expected.sha256}, computed ${actual}`,
    );
  }
  return bytes;
}
