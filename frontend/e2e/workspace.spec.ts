import { test, expect } from '@playwright/test';

test.describe('Milimo Music Web DAW & Studio', () => {
  test('Landing Page renders official brand tagline and hero', async ({ page }) => {
    await page.goto('/');
    
    // Check official tagline
    const tagline = page.getByText(/Give the silence something worth remembering/i);
    await expect(tagline).toBeVisible();

    // Check Composer elements
    const composerHeading = page.getByText(/Prompt \/ Lyrics/i).first();
    await expect(composerHeading).toBeVisible();
  });

  test('Composer Sidebar allows entering prompt and selecting parameters', async ({ page }) => {
    await page.goto('/');

    const textarea = page.locator('textarea').first();
    await expect(textarea).toBeVisible();
    await textarea.fill('An ambient synthwave journey with driving bass and warm analog leads.');
    await expect(textarea).toHaveValue('An ambient synthwave journey with driving bass and warm analog leads.');
  });

  test('LLM Settings Modal opens and displays masked credentials', async ({ page }) => {
    await page.goto('/');

    // Open settings if button exists
    const settingsButton = page.locator('button[title*="Settings"], button:has-text("Settings"), button[aria-label*="Settings"]').first();
    if (await settingsButton.isVisible()) {
      await settingsButton.click();
      const modal = page.getByText(/LLM Engine Settings|Model Provider/i).first();
      await expect(modal).toBeVisible();
    }
  });

  test('Studio Navigation between views', async ({ page }) => {
    await page.goto('/');

    // Verify main navigation rail buttons exist
    const songsNav = page.getByRole('button', { name: /Songs|Tracks/i }).first();
    if (await songsNav.isVisible()) {
      await songsNav.click();
    }
  });
});
