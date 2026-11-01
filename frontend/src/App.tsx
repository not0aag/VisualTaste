/**
 * The shell: a nav bar and the six pages.
 */

import { NavLink, Navigate, Route, Routes, BrowserRouter } from 'react-router-dom'

import { EmbeddingMap } from './pages/EmbeddingMap'
import { Feed } from './pages/Feed'
import { History } from './pages/History'
import { ModelLab } from './pages/ModelLab'
import { Profile } from './pages/Profile'
import { Review } from './pages/Review'

const PAGES = [
  { path: '/feed', label: 'Feed' },
  { path: '/history', label: 'History' },
  { path: '/profile', label: 'Profile' },
  { path: '/lab', label: 'Model Lab' },
  { path: '/map', label: 'Embedding Map' },
  { path: '/review', label: 'Review' },
]

export function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full flex-col">
        <nav className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-canvas-300 bg-canvas-100 px-3 py-1.5 dark:border-canvas-800 dark:bg-canvas-950">
          <span className="mr-3 shrink-0 text-sm font-semibold tracking-tight">
            VisualTaste
          </span>
          {PAGES.map((page) => (
            <NavLink
              key={page.path}
              to={page.path}
              className={({ isActive }) =>
                `shrink-0 rounded px-2 py-1 text-xs transition-colors ${
                  isActive
                    ? 'bg-canvas-300 text-canvas-900 dark:bg-canvas-800 dark:text-canvas-100'
                    : 'text-canvas-500 hover:text-canvas-800 dark:hover:text-canvas-200'
                }`
              }
            >
              {page.label}
            </NavLink>
          ))}
        </nav>

        <div className="min-h-0 flex-1">
          <Routes>
            <Route path="/" element={<Navigate to="/feed" replace />} />
            <Route path="/feed" element={<Feed />} />
            <Route path="/history" element={<History />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/lab" element={<ModelLab />} />
            <Route path="/map" element={<EmbeddingMap />} />
            <Route path="/review" element={<Review />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}
