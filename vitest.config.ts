import { defineConfig } from 'vitest/config';

const shared = {
  restoreMocks: true,
  clearMocks: true,
} as const;

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          ...shared,
          name: 'node',
          include: ['tests/unit/**/*.test.ts'],
          exclude: ['tests/unit/dom/**'],
          environment: 'node',
        },
      },
      {
        test: {
          ...shared,
          name: 'dom',
          include: ['tests/unit/dom/**/*.test.ts'],
          environment: 'jsdom',
        },
      },
    ],
  },
});
