import { describe, expect, it } from 'vitest';
import { webcrypto } from 'node:crypto';
import { IntegrityError, sha256Hex, verifyArtifact } from '../../src/shared/integrity.ts';

const subtle = webcrypto.subtle;
const BYTES = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
// sha256 of the ten bytes above.
const DIGEST = '1f825aa2f0020ef7cf91dfa30da4668d791c5d4824fc8e41354b89ec05795ab3';

describe('sha256Hex', () => {
  it('produces lowercase hex for a known vector', async () => {
    expect(await sha256Hex(BYTES, subtle)).toBe(DIGEST);
  });

  it('produces the empty-input vector', async () => {
    expect(await sha256Hex(new Uint8Array(0), subtle)).toBe(
      'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    );
  });
});

describe('verifyArtifact', () => {
  const expected = { sizeBytes: BYTES.byteLength, sha256: DIGEST };

  it('returns the verified bytes when size and digest both match', async () => {
    await expect(verifyArtifact(BYTES, expected, subtle)).resolves.toBe(BYTES);
  });

  it('rejects on a size mismatch before hashing anything', async () => {
    let hashCalls = 0;
    const countingSubtle = {
      digest: (alg: 'SHA-256', data: BufferSource) => {
        hashCalls += 1;
        return subtle.digest(alg, data);
      },
    };
    await expect(
      verifyArtifact(BYTES, { ...expected, sizeBytes: 11 }, countingSubtle),
    ).rejects.toThrow(IntegrityError);
    expect(hashCalls).toBe(0);
  });

  it('rejects on a digest mismatch and names both digests', async () => {
    const wrong = 'f'.repeat(64);
    await expect(verifyArtifact(BYTES, { ...expected, sha256: wrong }, subtle)).rejects.toThrow(
      new RegExp(`${wrong}[\\s\\S]*${DIGEST}|${DIGEST}[\\s\\S]*${wrong}`),
    );
  });

  it('reports which check failed on the error', async () => {
    const err = await verifyArtifact(BYTES, { ...expected, sizeBytes: 9 }, subtle).catch(
      (e: unknown) => e as IntegrityError,
    );
    expect(err).toBeInstanceOf(IntegrityError);
    expect((err as IntegrityError).check).toBe('size');
  });
});
