/**
 * One photograph, draggable, framed like a HUD target.
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

import { animate, motion, useMotionValue, useTransform, type MotionValue } from 'framer-motion'
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
        <motion.div className="relative p-3" style={{ x, y, rotate }}>
          {/* HUD corner brackets around the frame. */}
          <Corner className="left-0 top-0 border-l-2 border-t-2" />
          <Corner className="right-0 top-0 border-r-2 border-t-2" />
          <Corner className="bottom-0 left-0 border-b-2 border-l-2" />
          <Corner className="bottom-0 right-0 border-b-2 border-r-2" />

          <motion.img
            key={image.image_id}
            src={image.display_url}
            alt={image.attribution.title ?? 'An image to rate'}
            draggable={false}
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            className="max-h-[70vh] max-w-full rounded-lg object-contain shadow-card ring-1 ring-white/10"
          />

          {/* Verdict tints and stamps. `pointer-events-none` so they never eat the drag. */}
          <Verdict opacity={likeOpacity} tint="from-like/40" text="Like" textClass="right-6 top-6 rotate-6 border-like text-like" />
          <Verdict opacity={dislikeOpacity} tint="from-dislike/40" text="Nope" textClass="left-6 top-6 -rotate-6 border-dislike text-dislike" />
          <Verdict opacity={saveOpacity} tint="from-save/40" text="Save" textClass="bottom-8 left-1/2 -translate-x-1/2 border-save text-save" />
        </motion.div>
      </div>

      {showExplanation && (
        // Centred by a flex wrapper, not a transform: the fade-in animation
        // sets `transform` and would undo a translate-based centring.
        <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-4">
          <div
            data-testid="explanation-panel"
            className="panel pointer-events-auto w-full max-w-lg animate-fade-up border-accent/30 px-4 py-3 text-sm shadow-glow"
          >
            <p className="eyebrow mb-1">Why this image</p>
            <p className="font-medium text-white">{card.sentence}</p>
            <p className="mt-2 flex flex-wrap gap-2 text-xs">
              <span className="chip">
                reason <span className="text-accent">{card.reason}</span>
              </span>
              <span className="chip">
                score <span className="text-accent">{card.score.toFixed(3)}</span>
              </span>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function Corner({ className }: { className: string }) {
  return (
    <span
      aria-hidden
      className={`pointer-events-none absolute h-6 w-6 border-accent/70 drop-shadow-[0_0_6px_rgba(34,211,238,0.8)] ${className}`}
    />
  )
}

function Verdict({
  opacity,
  tint,
  text,
  textClass,
}: {
  opacity: MotionValue<number>
  tint: string
  text: string
  textClass: string
}) {
  return (
    <motion.div style={{ opacity }} className="pointer-events-none absolute inset-3 overflow-hidden rounded-lg">
      <div className={`absolute inset-0 bg-gradient-to-t ${tint} to-transparent`} />
      <span
        className={`absolute rounded-md border-2 bg-canvas-950/60 px-3 py-1 font-mono text-2xl font-bold uppercase tracking-[0.25em] backdrop-blur-sm ${textClass}`}
      >
        {text}
      </span>
    </motion.div>
  )
}
