/**
 * One photograph, draggable, on neutral grey.
 *
 * The gesture rules come from brief section 7.9: right is like, left is
 * dislike, up is save, and a gesture commits at 30% of the viewport width or
 * on a fast flick. The card tilts and tints while dragging so the verdict is
 * legible before the finger lifts -- without that feedback a drag feels like
 * nothing is happening until it suddenly does.
 *
 * Why a velocity threshold as well as a distance one: a short, fast flick is
 * how people actually swipe once they are used to an interface, and requiring
 * 30% of the screen every time makes a familiar app feel heavy.
 *
 * The image is `object-contain` on a neutral background, never cropped. The
 * corpus holds everything from 1:1 museum scans to 3:2 photographs, and
 * cropping to fill would hide part of what is being judged.
 */

import { animate, motion, useMotionValue, useTransform } from 'framer-motion'
import { useDrag } from '@use-gesture/react'
import { useEffect } from 'react'

import type { Action, FeedCard } from '../api/types'

/** Fraction of the viewport width a drag must cross to commit. */
const COMMIT_FRACTION = 0.3

/** Pixels per millisecond past which a flick commits regardless of distance. */
const FLICK_VELOCITY = 0.5

/** Degrees of tilt at full commit distance. */
const MAX_TILT = 12

export interface SwipeCardProps {
  card: FeedCard
  onCommit: (action: Action) => void
  showExplanation: boolean
  disabled?: boolean
}

export function SwipeCard({ card, onCommit, showExplanation, disabled }: SwipeCardProps) {
  const x = useMotionValue(0)
  const y = useMotionValue(0)

  const commitDistance = () =>
    (typeof window === 'undefined' ? 400 : window.innerWidth) * COMMIT_FRACTION

  const rotate = useTransform(x, [-600, 0, 600], [-MAX_TILT, 0, MAX_TILT])
  const likeOpacity = useTransform(x, [0, 160], [0, 0.85])
  const dislikeOpacity = useTransform(x, [-160, 0], [0.85, 0])
  const saveOpacity = useTransform(y, [-160, 0], [0.85, 0])

  // A new card starts centred, however the previous one left.
  useEffect(() => {
    x.set(0)
    y.set(0)
  }, [card.image.image_id, x, y])

  const bind = useDrag(
    ({ down, movement: [mx, my], velocity: [vx, vy], direction: [dx, dy] }) => {
      if (disabled) return

      if (down) {
        x.set(mx)
        y.set(my)
        return
      }

      const threshold = commitDistance()
      const horizontalFlick = vx > FLICK_VELOCITY
      const verticalFlick = vy > FLICK_VELOCITY

      // Up beats sideways: a save is the rarer, more deliberate gesture, so an
      // ambiguous diagonal should not silently become a like.
      if ((-my > threshold || (verticalFlick && dy < 0)) && -my > Math.abs(mx)) {
        void animate(y, -900, { duration: 0.25 })
        onCommit('save')
        return
      }
      if (mx > threshold || (horizontalFlick && dx > 0)) {
        void animate(x, 900, { duration: 0.25 })
        onCommit('like')
        return
      }
      if (-mx > threshold || (horizontalFlick && dx < 0)) {
        void animate(x, -900, { duration: 0.25 })
        onCommit('dislike')
        return
      }

      // Not far enough: spring back, so an accidental nudge is not a verdict.
      void animate(x, 0, { type: 'spring', stiffness: 400, damping: 36 })
      void animate(y, 0, { type: 'spring', stiffness: 400, damping: 36 })
    },
    { filterTaps: true, pointer: { touch: true } },
  )

  const { image } = card

  return (
    <div className="relative flex h-full w-full items-center justify-center overflow-hidden">
      {/* The gesture target is a plain element and the animated one sits
          inside it. Both libraries define an `onDrag` prop with incompatible
          signatures, so spreading @use-gesture's bindings onto a `motion.div`
          does not type-check -- and casting it away would hide a real
          mismatch rather than resolve it. */}
      <div
        {...bind()}
        role="group"
        aria-label="Image to rate. Drag right to like, left to dislike, up to save."
        data-testid="swipe-card"
        data-image-id={image.image_id}
        className="relative max-h-full max-w-full cursor-grab touch-none select-none active:cursor-grabbing"
      >
      <motion.div className="relative" style={{ x, y, rotate }}>
        <img
          src={image.display_url}
          alt={image.attribution.title ?? 'An image to rate'}
          draggable={false}
          className="max-h-[78vh] max-w-full rounded-sm object-contain shadow-2xl"
        />

        {/* Verdict tints. `pointer-events-none` so they never eat the drag. */}
        <motion.div
          style={{ opacity: likeOpacity }}
          className="pointer-events-none absolute inset-0 rounded-sm bg-like/30"
        />
        <motion.div
          style={{ opacity: dislikeOpacity }}
          className="pointer-events-none absolute inset-0 rounded-sm bg-dislike/30"
        />
        <motion.div
          style={{ opacity: saveOpacity }}
          className="pointer-events-none absolute inset-0 rounded-sm bg-save/30"
        />
      </motion.div>
      </div>

      {showExplanation && (
        <div
          data-testid="explanation-panel"
          className="absolute bottom-4 left-1/2 max-w-lg -translate-x-1/2 rounded-md bg-canvas-900/90 px-4 py-3 text-sm text-canvas-100 shadow-lg"
        >
          <p className="font-medium">{card.sentence}</p>
          <p className="mt-1 text-xs text-canvas-400">
            reason <span className="font-mono">{card.reason}</span> · score{' '}
            <span className="font-mono">{card.score.toFixed(3)}</span>
          </p>
        </div>
      )}
    </div>
  )
}
