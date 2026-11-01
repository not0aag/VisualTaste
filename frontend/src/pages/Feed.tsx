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

import { useCallback, useEffect, useRef, useState } from 'react'

import { AttributionLine } from '../components/AttributionLine'
import {
  AmbientLight,
  AnimatedNumber,
  Centred,
  EdgeFlash,
  LearningRing,
  Spinner,
  VerdictButton,
} from '../components/FeedParts'
import { SwipeCard } from '../components/SwipeCard'
import { useDwellTimer } from '../hooks/useDwellTimer'
import { useFeedQueue } from '../hooks/useFeedQueue'
import type { Action } from '../api/types'

const ANNOUNCE: Record<Action, string> = {
  like: 'Liked',
  dislike: 'Disliked',
  save: 'Saved',
  skip: 'Skipped',
}

const SHORTCUTS: [string, string][] = [
  ['→  or  L', 'Like'],
  ['←  or  J', 'Dislike'],
  ['↓  or  Space', 'Skip'],
  ['S', 'Save'],
  ['Z  or  Backspace', 'Undo the last verdict'],
  ['I', 'Why this image'],
  ['O', 'Open the original page'],
  ['?', 'Show or hide this list'],
]

/** Every shortcut on one card; `?` or Escape closes it. */
function ShortcutOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="absolute inset-0 z-40 flex items-center justify-center bg-canvas-950/70 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-label="Keyboard shortcuts"
      data-testid="shortcut-overlay"
    >
      <div className="plate w-full max-w-sm animate-fade-up p-5" onClick={(e) => e.stopPropagation()}>
        <p className="eyebrow mb-3">Keyboard</p>
        <dl className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-2 text-sm">
          {SHORTCUTS.map(([keys, action]) => (
            <div key={action} className="contents">
              <dt className="font-mono text-xs text-accent">{keys}</dt>
              <dd className="text-canvas-200">{action}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-canvas-500">Or drag the photo: right likes, left dislikes, up saves.</p>
      </div>
    </div>
  )
}

/** Labels between online retrains (backend `VT_RETRAIN_EVERY`, config.py). */
const RETRAIN_EVERY = 20


export function Feed() {
  const feed = useFeedQueue()
  const { current } = feed
  const [showExplanation, setShowExplanation] = useState(false)
  // The on-screen key that echoes a keystroke, and the edge flash per verdict.
  const [pressed, setPressed] = useState<Action | 'undo' | null>(null)
  const [flash, setFlash] = useState<{ action: Action; n: number } | null>(null)
  const [showKeys, setShowKeys] = useState(false)
  // Read out to screen readers after each verdict (an aria-live region).
  const [announcement, setAnnouncement] = useState('')
  const pressTimer = useRef<number>()

  const echo = useCallback((key: Action | 'undo') => {
    setPressed(key)
    window.clearTimeout(pressTimer.current)
    pressTimer.current = window.setTimeout(() => setPressed(null), 170)
  }, [])
  useEffect(() => () => window.clearTimeout(pressTimer.current), [])

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
      echo(action)
      setFlash((previous) => ({ action, n: (previous?.n ?? 0) + 1 }))
      setAnnouncement(`${ANNOUNCE[action]}. Next image.`)
      void feed.swipe(action, dwellMs)
    },
    [current, dwell, feed, echo],
  )

  const undo = useCallback(() => {
    // `lastSwipe` is cleared on success, so this cannot undo twice.
    if (!feed.lastSwipe) return
    echo('undo')
    setAnnouncement('Undone. The previous image is back.')
    void feed.undo()
  }, [feed, echo])

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
        case '?':
          event.preventDefault()
          setShowKeys((shown) => !shown)
          break
        case 'Escape':
          setShowKeys(false)
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
        <p className="mt-2 font-display text-xl font-semibold uppercase tracking-wide text-white">
          The collection is still being built
        </p>
        <p className="mt-2 max-w-md text-sm text-canvas-400">{feed.error}</p>
      </Centred>
    )
  }

  if (feed.exhausted || !current) {
    return (
      <Centred>
        <p className="eyebrow">Complete</p>
        <p className="mt-2 font-display text-xl font-semibold uppercase tracking-wide text-white">
          That is every image in the collection
        </p>
        <p className="mt-2 text-sm text-canvas-400">
          {feed.labels.toLocaleString()} rated. Nothing left to show.
        </p>
      </Centred>
    )
  }

  const untilUpdate = RETRAIN_EVERY - (feed.labels % RETRAIN_EVERY)

  return (
    <div className="relative isolate flex h-full flex-col overflow-hidden">
      <AmbientLight colours={current.image.dominant_colours} imageId={current.image.image_id} />
      <EdgeFlash flash={flash} />

      <header className="flex items-center justify-between gap-4 px-4 pt-4 sm:px-6">
        <span className="flex items-baseline gap-2" data-testid="label-count">
          <AnimatedNumber
            value={feed.labels}
            className="font-display text-3xl font-semibold leading-none tabular-nums text-white"
          />
          <span className="eyebrow">rated</span>
        </span>

        <span className="flex items-center gap-3">
          <LearningRing
            done={RETRAIN_EVERY - untilUpdate}
            total={RETRAIN_EVERY}
            label={
              feed.retrainQueued
                ? 'Model updating with your latest ratings'
                : `Model update in ${untilUpdate} rating${untilUpdate === 1 ? '' : 's'}`
            }
            busy={feed.retrainQueued}
          />
          <span className="chip" title={`model version ${feed.modelVersion}`}>
            <span className="status-dot" aria-hidden />
            <span className="uppercase">{feed.rung}</span>
            {feed.retrainQueued && (
              <span className="ml-1 text-accent" data-testid="retraining">
                updating…
              </span>
            )}
          </span>
        </span>
      </header>

      <p className="sr-only" aria-live="polite" data-testid="announcer">
        {announcement}
      </p>
      {showKeys && <ShortcutOverlay onClose={() => setShowKeys(false)} />}

      <main className="flex min-h-0 flex-1 items-start justify-center px-3 pt-4 sm:items-center sm:py-2">
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
          <VerdictButton onClick={() => commit('dislike')} testId="btn-dislike" tone="dislike" keyHint="←" pressed={pressed === 'dislike'}>
            Dislike
          </VerdictButton>
          <VerdictButton onClick={() => commit('skip')} testId="btn-skip" keyHint="↓" pressed={pressed === 'skip'}>
            Skip
          </VerdictButton>
          <VerdictButton onClick={() => commit('save')} testId="btn-save" tone="save" keyHint="S" pressed={pressed === 'save'}>
            Save
          </VerdictButton>
          <VerdictButton onClick={() => commit('like')} testId="btn-like" tone="like" keyHint="→" pressed={pressed === 'like'}>
            Like
          </VerdictButton>
          <span className="mx-1 hidden h-6 w-px bg-white/10 sm:block" aria-hidden />
          <VerdictButton
            onClick={() => undo()}
            testId="btn-undo"
            disabled={!feed.lastSwipe}
            keyHint="Z"
            pressed={pressed === 'undo'}
          >
            Undo
          </VerdictButton>
        </div>

        <p className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-canvas-500">
          <span>
            <span className="kbd">I</span> why this image
          </span>
          <span>
            <span className="kbd">O</span> open source
          </span>
          <span>
            <span className="kbd">Space</span> skip
          </span>
          <span>
            <span className="kbd">?</span> all keys
          </span>
          <span className="hidden sm:inline">or drag the photo: right likes, left dislikes, up saves</span>
        </p>
      </footer>
    </div>
  )
}
