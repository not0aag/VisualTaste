/**
 * The shell: the instrument's top bar and the six pages.
 */

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { NavLink, Navigate, Route, Routes, BrowserRouter, useLocation } from 'react-router-dom'

import { DEMO } from './api/client'
import { EmbeddingMap } from './pages/EmbeddingMap'
import { Feed } from './pages/Feed'
import { History } from './pages/History'
import { ModelLab } from './pages/ModelLab'
import { Profile } from './pages/Profile'
import { Review } from './pages/Review'

/** 16px line icons, drawn on a 24-unit grid with one stroke weight. */
const ICONS: Record<string, JSX.Element> = {
  // A viewfinder: the one photograph being judged.
  feed: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><circle cx="12" cy="12" r="3" /></>,
  // A contact sheet.
  history: <><rect x="4" y="4" width="7" height="7" rx="1" /><rect x="13" y="4" width="7" height="7" rx="1" /><rect x="4" y="13" width="7" height="7" rx="1" /><rect x="13" y="13" width="7" height="7" rx="1" /></>,
  // Interests as overlapping clusters.
  profile: <><circle cx="9" cy="10" r="4.5" /><circle cx="15.5" cy="14" r="4.5" /></>,
  // A learning curve.
  lab: <><path d="M4 19h16" /><path d="M5 16c3-1 4-8 7-8s3.5 4 7-4" /></>,
  // Scattered points in a latent space.
  map: <><circle cx="6" cy="8" r="1.3" /><circle cx="10" cy="14" r="1.3" /><circle cx="15" cy="7" r="1.3" /><circle cx="18" cy="15" r="1.3" /><circle cx="8" cy="18" r="1.3" /><circle cx="13" cy="11" r="1.3" /></>,
  // A shield: the safety screen.
  review: <><path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6z" /><path d="m9.5 12 2 2 3.5-4" /></>,
}

const PAGES = [
  { path: '/feed', label: 'Feed', icon: 'feed' },
  { path: '/history', label: 'History', icon: 'history' },
  { path: '/profile', label: 'Profile', icon: 'profile' },
  { path: '/lab', label: 'Model Lab', icon: 'lab' },
  { path: '/map', label: 'Embedding Map', icon: 'map' },
  { path: '/review', label: 'Review', icon: 'review' },
]

export function App() {
  return (
    // On GitHub Pages the app lives under /VisualTaste/; locally under /.
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '') || '/'}>
      <Shell />
    </BrowserRouter>
  )
}

function Shell() {
  const location = useLocation()
  const reduce = useReducedMotion()

  return (
    <div className="flex h-full flex-col">
      {DEMO && <DemoBanner />}
      <nav className="relative z-20 flex shrink-0 items-center gap-0.5 overflow-x-auto border-b border-white/[0.06] bg-canvas-950/75 px-3 py-2 backdrop-blur-xl sm:px-5">
        <span className="mr-5 flex shrink-0 items-center gap-2.5">
          <LogoMark />
          <span className="font-display text-[15px] font-semibold uppercase tracking-[0.18em] text-white">
            Visual<span className="text-accent">Taste</span>
          </span>
        </span>

        {PAGES.map((page) => (
          <NavLink
            key={page.path}
            to={page.path}
            className={({ isActive }) =>
              `relative flex shrink-0 items-center gap-2 rounded-plate px-3 py-1.5 text-[13px] transition-colors ${
                isActive ? 'text-white' : 'text-canvas-400 hover:text-canvas-100'
              }`
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  // One indicator, shared across tabs, so it slides between them.
                  <motion.span
                    layoutId="nav-active"
                    transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 500, damping: 40 }}
                    className="absolute inset-0 rounded-plate bg-accent/10 shadow-[inset_0_0_0_1px_rgba(63,216,240,0.35)]"
                  >
                    <span className="absolute inset-x-3 -bottom-[9px] h-px bg-gradient-to-r from-transparent via-accent to-transparent" />
                  </motion.span>
                )}
                <svg
                  viewBox="0 0 24 24"
                  className={`relative h-4 w-4 fill-none stroke-current stroke-[1.6] ${isActive ? 'text-accent' : ''}`}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  {ICONS[page.icon]}
                </svg>
                <span className="relative">{page.label}</span>
              </>
            )}
          </NavLink>
        ))}

        <span
          className="ml-auto hidden shrink-0 items-center gap-2 pl-4 font-mono text-[10px] uppercase tracking-[0.2em] text-canvas-400 md:flex"
          title="Everything runs on this machine; nothing leaves it."
        >
          <span className="status-dot" aria-hidden />
          local
        </span>
      </nav>

      <div className="relative min-h-0 flex-1">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={location.pathname}
            className="h-full"
            initial={reduce ? false : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? { opacity: 1 } : { opacity: 0, y: -4 }}
            transition={{ duration: 0.18, ease: 'easeOut' }}
          >
            <Routes location={location}>
              <Route path="/" element={<Navigate to="/feed" replace />} />
              <Route path="/feed" element={<Feed />} />
              <Route path="/history" element={<History />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="/lab" element={<ModelLab />} />
              <Route path="/map" element={<EmbeddingMap />} />
              <Route path="/review" element={<Review />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}

/** Says plainly what the Pages demo is, on every page. */
function DemoBanner() {
  return (
    <div
      role="status"
      className="flex shrink-0 flex-wrap items-center justify-center gap-x-3 gap-y-1 border-b border-violet/30 bg-violet/10 px-4 py-1.5 text-center text-xs text-canvas-200"
    >
      <span className="eyebrow text-violet">Demo</span>
      <span>
        A replay of a recorded session with a simulated user, running entirely in your
        browser. Swipe away; nothing is sent anywhere.
      </span>
      <a className="link" href="https://github.com/not0aag/VisualTaste#quick-start">
        Run the real thing locally
      </a>
    </div>
  )
}

/** An eye inside a hexagon: taste, seen by a machine. */
function LogoMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-6 w-6 drop-shadow-[0_0_6px_rgba(63,216,240,0.6)]" aria-hidden>
      <defs>
        <linearGradient id="vt-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#3fd8f0" />
          <stop offset="1" stopColor="#a594f9" />
        </linearGradient>
      </defs>
      <path d="M16 2 28 9v14l-12 7L4 23V9z" fill="none" stroke="url(#vt-logo)" strokeWidth="1.8" />
      <path d="M8 16s3-5 8-5 8 5 8 5-3 5-8 5-8-5-8-5z" fill="none" stroke="url(#vt-logo)" strokeWidth="1.6" />
      <circle cx="16" cy="16" r="2.4" fill="#3fd8f0" />
    </svg>
  )
}
