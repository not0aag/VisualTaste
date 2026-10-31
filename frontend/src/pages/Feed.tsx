/**
 * The Feed: one image, full screen, judged with a flick or a keystroke.
 *
 * The keyboard map is brief section 7.9, and it exists because that is how
 * anyone labels a few thousand images without it becoming a chore:
 *
 *     -> or L     like
 *     <- or J     dislike
 *     v or Space  skip
 *     S           save
 *     Z or Back   undo
 *     I           toggle the explanation
 *     O           open the original page
 *
 * Note what is deliberately *not* here: no infinite scroll, no counts of how
 * many images remain, no streak. The page asks one question at a time, because
 * the quality of the labels depends on each one being an actual judgement.
 */

import { useCallback, useEffect, useState } from 'react'

import { AttributionLine } from '../components/AttributionLine'
import { SwipeCard } from '../components/SwipeCard'
import { useDwellTimer } from '../hooks/useDwellTimer'
import { useFeedQueue } from '../hooks/useFeedQueue'
import type { Action } from '../api/types'

export function Feed() {
  const feed = useFeedQueue()
  const { current } = feed
  const [showExplanation, setShowExplanation] = useState(false)
  const [busy, setBusy] = useState(false)

  const dwell = useDwellTimer(current?.image.image_id ?? null)

  const commit = useCallback(
    async (action: Action) => {
      if (!current || busy) return
      dwell.noteInteraction()
      const { dwellMs } = dwell.read()
      setBusy(true)
      try {
        await feed.swipe(action, dwellMs)
      } finally {
        setBusy(false)
      }
    },
    [current, busy, dwell, feed],
  )

  const undo = useCallback(async () => {
    if (busy) return
    setBusy(true)
    try {
      await feed.undo()
    } finally {
      setBusy(false)
    }
  }, [busy, feed])

  const openSource = useCallback(() => {
    const url = current?.image.attribution.source_url
    if (url) window.open(url, '_blank', 'noreferrer,noopener')
  }, [current])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Let the browser's own shortcuts through, and ignore keys typed into a
      // field (the History page's filters live in the same app).
      if (event.metaKey || event.ctrlKey || event.altKey) return
      const target = event.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return

      switch (event.key) {
        case 'ArrowRight':
        case 'l':
        case 'L':
          event.preventDefault()
          void commit('like')
          break
        case 'ArrowLeft':
        case 'j':
        case 'J':
          event.preventDefault()
          void commit('dislike')
          break
        case 'ArrowDown':
        case ' ':
          event.preventDefault()
          void commit('skip')
          break
        case 's':
        case 'S':
          event.preventDefault()
          void commit('save')
          break
        case 'z':
        case 'Z':
        case 'Backspace':
          event.preventDefault()
          void undo()
          break
        case 'i':
        case 'I':
          event.preventDefault()
          setShowExplanation((shown) => !shown)
          break
        case 'o':
        case 'O':
          event.preventDefault()
          openSource()
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [commit, undo, openSource])

  if (feed.loading) {
    return <Centred>Loading the feed…</Centred>
  }

  if (feed.stillBuilding) {
    return (
      <Centred>
        <p className="font-medium">The collection is still being built.</p>
        <p className="mt-2 max-w-md text-sm text-canvas-500">{feed.error}</p>
      </Centred>
    )
  }

  if (feed.exhausted || !current) {
    return (
      <Centred>
        <p className="font-medium">That is every image in the collection.</p>
        <p className="mt-2 text-sm text-canvas-500">
          {feed.labels.toLocaleString()} rated. Nothing left to show.
        </p>
      </Centred>
    )
  }

  return (
    <div className="flex h-full flex-col bg-canvas-200 dark:bg-canvas-900">
      <header className="flex items-center justify-between px-4 py-2 text-xs text-canvas-500 dark:text-canvas-400">
        <span data-testid="label-count">{feed.labels.toLocaleString()} rated</span>
        <span className="font-mono" title={`model version ${feed.modelVersion}`}>
          {feed.rung}
          {feed.retrainQueued && (
            <span className="ml-2 text-canvas-400" data-testid="retraining">
              updating…
            </span>
          )}
        </span>
      </header>

      <main className="flex min-h-0 flex-1 items-center justify-center px-2">
        <SwipeCard
          card={current}
          onCommit={(action) => void commit(action)}
          showExplanation={showExplanation}
          disabled={busy}
        />
      </main>

      <footer className="flex flex-col items-center gap-3 px-4 pb-5 pt-2">
        <AttributionLine attribution={current.image.attribution} />

        {feed.error && !feed.stillBuilding && (
          <p role="alert" className="text-xs text-dislike">
            {feed.error}
          </p>
        )}

        <div className="flex items-center gap-2">
          <Button onClick={() => void commit('dislike')} testId="btn-dislike" tone="dislike">
            Dislike
          </Button>
          <Button onClick={() => void commit('skip')} testId="btn-skip">
            Skip
          </Button>
          <Button onClick={() => void commit('save')} testId="btn-save" tone="save">
            Save
          </Button>
          <Button onClick={() => void commit('like')} testId="btn-like" tone="like">
            Like
          </Button>
          <Button
            onClick={() => void undo()}
            testId="btn-undo"
            disabled={!feed.lastSwipe}
          >
            Undo
          </Button>
        </div>

        <p className="text-[11px] text-canvas-400">
          ← dislike · → like · ↓ skip · S save · Z undo · I why · O source
        </p>
      </footer>
    </div>
  )
}

function Centred({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center bg-canvas-200 p-8 text-center text-canvas-700 dark:bg-canvas-900 dark:text-canvas-200">
      {children}
    </div>
  )
}

function Button({
  children,
  onClick,
  testId,
  tone,
  disabled,
}: {
  children: React.ReactNode
  onClick: () => void
  testId: string
  tone?: 'like' | 'dislike' | 'save'
  disabled?: boolean
}) {
  const toneClass =
    tone === 'like'
      ? 'border-like/40 text-like hover:bg-like/10'
      : tone === 'dislike'
        ? 'border-dislike/40 text-dislike hover:bg-dislike/10'
        : tone === 'save'
          ? 'border-save/40 text-save hover:bg-save/10'
          : 'border-canvas-300 text-canvas-600 hover:bg-canvas-300/40 dark:border-canvas-700 dark:text-canvas-300 dark:hover:bg-canvas-800'

  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border px-3 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${toneClass}`}
    >
      {children}
    </button>
  )
}
