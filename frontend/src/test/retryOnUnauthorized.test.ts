import { describe, it, expect, vi } from 'vitest';
import { ApiError, retryOnUnauthorized } from '../api/client';

describe('retryOnUnauthorized', () => {
  it('signs in again and retries once when the session is gone', async () => {
    const run = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(401, 'invalid_credentials', 'Authentication required'))
      .mockResolvedValueOnce('ok');
    const relogin = vi.fn().mockResolvedValue(undefined);

    await expect(retryOnUnauthorized(run, relogin)).resolves.toBe('ok');
    expect(relogin).toHaveBeenCalledTimes(1);
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('passes other errors through without signing in', async () => {
    const run = vi.fn().mockRejectedValue(new ApiError(422, 'invalid_input', 'Bad file'));
    const relogin = vi.fn();

    await expect(retryOnUnauthorized(run, relogin)).rejects.toThrow('Bad file');
    expect(relogin).not.toHaveBeenCalled();
  });
});
