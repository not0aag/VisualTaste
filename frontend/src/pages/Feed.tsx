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

  const dwell = useDwellTimer(current?.image.image_id ?? null)

  /**
   * Deliberately *not* gated on an in-flight request.
   *
   * An earlier version ignored input while a swipe was being saved, and the
   * end-to-end test caught what that costs: swiping at speed silently dropped
   * verdicts, stalling at 10 of 20. For someone labelling a few thousand
   * images that is a real defect, not a test artefact.
   *
   * It is safe to accept the next key immediately because the queue advances
   * optimistically and synchronously -- `feed.current` is already the next
   * card by the time this returns, so a fast second keystroke applies to that
   * card rather than double-rating this one. A failed save rolls its own card
   * back into the queue.
   */
  const commit = useCallback(
    (action: Action) => {
      if (!current) return
      dwell.noteInteraction()
      const { dwellMs } = dwell.read()
      void feed.swipe(action, dwellMs)
    },
    [current, dwell, feed],
  )

  const undo = useCallback(() => {
    // `lastSwipe` is cleared on success, so this cannot undo twice.
    if (!feed.lastSwipe) return
    void feed.undo()
  }, [feed])

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
          commit('like')
          break
        case 'ArrowLeft':
        case 'j':
        case 'J':
          event.preventDefault()
          commit('dislike')
          break
        case 'ArrowDown':
        case ' ':
          event.preventDefault()
          commit('skip')
          break
        case 's':
        case 'S':
          event.preventDefault()
          commit('save')
          break
        case 'z':
        case 'Z':
        case 'Backspace':
          event.preventDefault()
          undo()
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
    return (
      <Centred>
        <Spinner />
        <p className="eyebrow mt-5">Loading the feed</p>
      </Centred>
    )
  }

  if (feed.stillBuilding) {
    return (
      <Centred>
        <p className="eyebrow">Standby</p>
        <p className="mt-2 text-lg font-medium text-white">The collection is still being built.</p>
        <p className="mt-2 max-w-md text-sm text-canvas-400">{feed.error}</p>
      </Centred>
    )
  }

  if (feed.exhausted || !current) {
    return (
      <Centred>
        <p className="eyebrow">Complete</p>
        <p className="mt-2 text-lg font-medium text-white">That is every image in the collection.</p>
        <p className="mt-2 text-sm text-canvas-400">
          {feed.labels.toLocaleString()} rated. Nothing left to show.
        </p>
      </Centred>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between px-4 pt-3 sm:px-6">
        <span className="flex items-baseline gap-2" data-testid="label-count">
          <span className="font-mono text-xl font-semibold tabular-nums text-white">
            {feed.labels.toLocaleString()}
          </span>
          <span className="eyebrow">rated</span>
        </span>
        <span className="chip" title={`model version ${feed.modelVersion}`}>
          <span className="status-dot" aria-hidden />
          <span className="uppercase">{feed.rung}</span>
          {feed.retrainQueued && (
            <span className="ml-1 text-accent" data-testid="retraining">
              updating…
            </span>
          )}
        </span>
      </header>

      <main className="flex min-h-0 flex-1 items-center justify-center px-3 py-2">
        <SwipeCard card={current} onCommit={commit} showExplanation={showExplanation} />
      </main>

      <footer className="flex flex-col items-center gap-3 px-4 pb-5 pt-1">
        <AttributionLine attribution={current.image.attribution} />

        {feed.error && !feed.stillBuilding && (
          <p role="alert" className="font-mono text-xs text-dislike">
            {feed.error}
          </p>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
          <Button onClick={() => commit('dislike')} testId="btn-dislike" tone="dislike" keyHint="←">
            Dislike
          </Button>
          <Button onClick={() => commit('skip')} testId="btn-skip" keyHint="↓">
            Skip
          </Button>
          <Button onClick={() => commit('save')} testId="btn-save" tone="save" keyHint="S">
            Save
          </Button>
          <Button onClick={() => commit('like')} testId="btn-like" tone="like" keyHint="→">
            Like
          </Button>
          <span className="mx-1 hidden h-6 w-px bg-white/10 sm:block" aria-hidden />
          <Button onClick={() => undo()} testId="btn-undo" disabled={!feed.lastSwipe} keyHint="Z">
            Undo
          </Button>
        </div>

        <p className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[11px] text-canvas-500">
          <span>
            <span className="kbd">I</span> why this image
          </span>
          <span>
            <span className="kbd">O</span> open source
          </span>
          <span>
            <span className="kbd">Space</span> skip
          </span>
          <span className="hidden sm:inline">drag the card: → like · ← dislike · ↑ save</span>
        </p>
      </footer>
    </div>
  )
}

function Centred({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center p-8 text-center">
      <div className="panel flex animate-fade-up flex-col items-center px-10 py-8">{children}</div>
    </div>
  )
}

function Spinner() {
  return (
    <span className="relative inline-flex h-10 w-10" aria-hidden>
      <span className="absolute inset-0 rounded-full border border-white/10" />
      <span className="absolute inset-0 animate-spin-slow rounded-full border-2 border-transparent border-t-accent shadow-[0_0_14px_-2px_rgba(34,211,238,0.7)]" />
    </span>
  )
}

const TONES = {
  like: 'border-like/40 text-like hover:bg-like/15 hover:shadow-glow-like',
  dislike: 'border-dislike/40 text-dislike hover:bg-dislike/15 hover:shadow-glow-dislike',
  save: 'border-save/40 text-save hover:bg-save/15 hover:shadow-glow-save',
  plain: 'border-white/10 text-canvas-300 hover:border-accent/40 hover:bg-accent/10 hover:text-white',
}

function Button({
  children,
  onClick,
  testId,
  tone,
  disabled,
  keyHint,
}: {
  children: React.ReactNode
  onClick: () => void
  testId: string
  tone?: 'like' | 'dislike' | 'save'
  disabled?: boolean
  keyHint?: string
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      disabled={disabled}
      className={`group inline-flex items-center gap-2 rounded-xl border bg-canvas-900/70 px-4 py-2 text-sm font-medium backdrop-blur transition-all active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 disabled:shadow-none ${TONES[tone ?? 'plain']}`}
    >
      {keyHint && (
        <span className="kbd opacity-70 group-hover:opacity-100" aria-hidden>
          {keyHint}
        </span>
      )}
      {children}
    </button>
  )
}
