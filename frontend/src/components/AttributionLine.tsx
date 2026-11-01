/**
 * Credit for the image, on every card.
 *
 * Not a nicety. Open Images is CC BY 2.0, which requires attributing the
 * photographer; the Unsplash dataset terms ask for the photographer's name and
 * the photo page. So this component appears on every image the app displays,
 * and the backend refuses to store an image it cannot credit (see
 * `data/sources/openimages.py`).
 *
 * The licence is a link rather than a label so that the actual terms are one
 * click away, and the source page link is what makes an individual image's
 * provenance checkable -- which matters most for Megalith, where the licence
 * guarantee is collection-level (DECISIONS.md #8).
 */

import type { Attribution } from '../api/types'

const SOURCE_NAMES: Record<string, string> = {
  megalith: 'Megalith-10M (Flickr, public domain)',
  pd12m: 'PD12M',
  open_images: 'Open Images V7',
  unsplash_lite: 'Unsplash',
}

const LICENCE_NAMES: Array<[string, string]> = [
  ['publicdomain/zero', 'CC0'],
  ['publicdomain/mark', 'Public Domain Mark'],
  ['licenses/by/2.0', 'CC BY 2.0'],
  ['licenses/by/4.0', 'CC BY 4.0'],
  ['unsplash.com/license', 'Unsplash Licence'],
  ['megalith-10m', 'Public domain (collection)'],
]

function licenceName(url: string): string {
  const match = LICENCE_NAMES.find(([fragment]) => url.includes(fragment))
  return match ? match[1] : 'Licence'
}

export function AttributionLine({ attribution }: { attribution: Attribution }) {
  const sourceLabel = SOURCE_NAMES[attribution.source] ?? attribution.source

  return (
    <p
      data-testid="attribution"
      className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 font-mono text-[11px] text-canvas-400"
    >
      {attribution.author && (
        <span>
          {attribution.author_url ? (
            <a
              href={attribution.author_url}
              target="_blank"
              rel="noreferrer noopener"
              className="transition-colors hover:text-accent"
            >
              {attribution.author}
            </a>
          ) : (
            attribution.author
          )}
        </span>
      )}

      {attribution.author && <span aria-hidden className="text-canvas-600">{'//'}</span>}

      <a
        href={attribution.license_url}
        target="_blank"
        rel="noreferrer noopener"
        className="transition-colors hover:text-accent"
      >
        {licenceName(attribution.license_url)}
      </a>

      <span aria-hidden className="text-canvas-600">{'//'}</span>

      {attribution.source_url ? (
        <a
          href={attribution.source_url}
          target="_blank"
          rel="noreferrer noopener"
          className="transition-colors hover:text-accent"
          title="Open the original page (O)"
        >
          {sourceLabel}
        </a>
      ) : (
        <span>{sourceLabel}</span>
      )}
    </p>
  )
}
