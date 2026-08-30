import { test, expect } from '@playwright/test';

/**
 * Artist section E2E (Phase H / D1): create stepper → detail → cover →
 * gated produce. Backend is mocked via page.route so the UI flow is
 * verified deterministically without model dependencies.
 */

const API = 'http://localhost:8000';

const now = '2026-08-29T12:00:00';

const profile = {
  id: 'p1', project_id: null, name: 'Nalo Rivers',
  bio: 'Raised between two cities and a river of late-night radio.',
  lore_json: '{}', tags: 'indie folk, warm',
  cover_image_path: null as string | null,
  default_provider: null, default_model: null,
  created_at: now, updated_at: now,
};

const release = {
  id: 'r1', profile_id: 'p1', title: 'First Light', description: '',
  status: 'planned', vision_json: '{}', created_at: now, updated_at: now,
};

const profileDetail = { profile, assignments: [], releases: [release] };

const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  'base64',
);

async function mockApi(page: import('@playwright/test').Page) {
  await page.route(`${API}/projects`, r => r.fulfill({ json: [] }));
  await page.route(`${API}/styles`, r => r.fulfill({ json: { styles: [] } }));
  await page.route(`${API}/agents`, r => r.fulfill({ json: { agents: [] } }));
  await page.route(`${API}/agents/runs*`, r => r.fulfill({ json: { runs: [], total: 0 } }));
  await page.route(`${API}/profiles?*`, r => r.fulfill({ json: { profiles: [profile], total: 1, stats: { p1: { crew_count: 2, release_count: 1, last_activity: now } } } }));
  await page.route(`${API}/profiles`, r => r.fulfill({ json: { profiles: [profile], total: 1, stats: { p1: { crew_count: 2, release_count: 1, last_activity: now } } } }));
  await page.route(`${API}/profiles/p1`, r => r.fulfill({ json: profileDetail }));
  await page.route(`${API}/releases/r1/tracks`, r => r.fulfill({
    json: {
      release_id: 'r1', title: 'First Light', status: 'in_progress', rollup: 'partial',
      tracks: [
        {
          id: 'j1', title: 'Side A', status: 'completed', duration_ms: 120000,
          seed: 42, seed_slot: 0,
          artifacts: { audio: '/audio/j1.wav', midi: null, musicxml: null, stems: null, mastered: null },
          used_real_inference: true, created_at: now,
        },
        {
          id: 'j2', title: 'Side B', status: 'failed', duration_ms: 0,
          seed: null, seed_slot: 1,
          artifacts: { audio: null, midi: null, musicxml: null, stems: null, mastered: null },
          used_real_inference: false, created_at: now,
        },
      ], succeeded: 1, total: 2,
    },
  }));
  await page.route(`${API}/upload/image`, r => r.fulfill({ json: { url: '/image/test-cover.png', filename: 'test-cover.png' } }));
  await page.route(`${API}/profiles/p1/cover`, r =>
    r.fulfill({ json: { ...profile, cover_image_path: '/image/test-cover.png' } }));
}

test.describe('Artist section', () => {
  test('empty state offers a create CTA and the guided stepper opens', async ({ page }) => {
    await page.route(`${API}/projects`, r => r.fulfill({ json: [] }));
    await page.route(`${API}/styles`, r => r.fulfill({ json: { styles: [] } }));
    await page.route(`${API}/agents`, r => r.fulfill({ json: { agents: [] } }));
    await page.route(`${API}/profiles?*`, r => r.fulfill({ json: { profiles: [], total: 0 } }));
    await page.route(`${API}/profiles`, r => r.fulfill({ json: { profiles: [], total: 0 } }));
    await page.goto('/?view=artists');

    await expect(page.getByRole('button', { name: /create your first artist/i })).toBeVisible();
    await page.getByRole('button', { name: /create your first artist/i }).click();

    // Stepper: step 1 — identity; Next disabled until the name is valid
    const dialog = page.getByRole('dialog', { name: /new artist profile/i });
    await expect(dialog).toBeVisible();
    const next = dialog.getByRole('button', { name: 'Next' });
    await expect(next).toBeDisabled();
    await dialog.getByLabel(/artist name/i).fill('Nalo Rivers');
    await expect(next).toBeEnabled();
    await next.click();
    // Step 2 — bio with char counter
    await expect(dialog.getByText(/bio \/ identity/i)).toBeVisible();
  });

  test('guided create lands on the new artist detail', async ({ page }) => {
    await mockApi(page);
    await page.route(`${API}/profiles`, async (r) => {
      if (r.request().method() === 'POST') {
        const body = r.request().postDataJSON();
        await r.fulfill({ json: { ...profile, name: body?.name ?? 'New Artist', tags: body?.tags ?? '' } });
      } else {
        await r.fulfill({ json: { profiles: [profile], total: 1 } });
      }
    });
    await page.goto('/?view=artists');
    await page.getByRole('button', { name: /new artist/i }).click();

    const dialog = page.getByRole('dialog', { name: /new artist profile/i });
    await dialog.getByLabel(/artist name/i).fill('Nalo Rivers');
    await dialog.getByRole('button', { name: 'Next' }).click();
    await dialog.getByRole('button', { name: 'Next' }).click();
    await dialog.getByRole('button', { name: 'Next' }).click();
    await dialog.getByRole('button', { name: /create artist/i }).click();

    // Detail opens automatically with the created artist
    await expect(page.getByRole('heading', { name: /nalo rivers/i })).toBeVisible();
  });

  test('detail shows release rows with gated produce and autopilot off by default', async ({ page }) => {
    await mockApi(page);
    await page.route(`${API}/jobs/j1`, r => r.fulfill({
      json: { id: 'j1', audio_path: '/audio/j1.wav', title: 'Side A', status: 'completed' },
    }));
    await page.goto('/?view=artists&id=p1');

    // Deep-link landed on the artist
    await expect(page.getByRole('heading', { name: /nalo rivers/i })).toBeVisible();
    await expect(page.getByText('First Light')).toBeVisible();
    // Produce present; autopilot exists and is unchecked (gated default)
    await expect(page.getByRole('button', { name: 'Produce', exact: true })).toBeVisible();
    const autopilot = page.getByLabel(/autopilot mode/i);
    await expect(autopilot).toBeVisible();
    await expect(autopilot).not.toBeChecked();
    // Budget cap selector ships with the produce controls
    await expect(page.getByLabel(/budget cap/i)).toBeVisible();

    // Tracklist: completed row offers playback + studio handoff, failed row offers retry
    await page.getByRole('button', { name: 'Tracks', exact: true }).first().click();
    await expect(page.getByRole('button', { name: '▶ Play' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Studio', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible();
    // Reorder controls
    await expect(page.getByRole('button', { name: /move side a up/i })).toBeDisabled();
    await expect(page.getByRole('button', { name: /move side a down/i })).toBeEnabled();
  });

  test('cover upload flows through upload → setCover and renders', async ({ page }) => {
    await mockApi(page);
    await page.goto('/?view=artists&id=p1');
    await expect(page.getByRole('heading', { name: /nalo rivers/i })).toBeVisible();

    const [chooser] = await Promise.all([
      page.waitForEvent('filechooser'),
      page.getByText(/add identity image/i).click(),
    ]);
    await chooser.setFiles({ name: 'cover.png', mimeType: 'image/png', buffer: PNG });
    await expect(page.getByText(/artist identity image updated/i)).toBeVisible();
  });

  test('list cards render stats and search filters client-side', async ({ page }) => {
    await mockApi(page);
    await page.goto('/?view=artists');
    await expect(page.getByText('2 crew')).toBeVisible();
    await expect(page.getByText('1 releases')).toBeVisible();

    const search = page.getByLabel('Search artists');
    await search.fill('zzz-no-match');
    await expect(page.getByText(/no artists match/i)).toBeVisible();
  });
});
