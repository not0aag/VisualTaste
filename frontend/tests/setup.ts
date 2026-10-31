import '@testing-library/jest-dom/vitest'

// jsdom has no layout, so `window.innerWidth` is 1024 by default. The swipe
// threshold is a fraction of it, which is fine -- but tests that simulate
// gestures set explicit values rather than relying on this.

// `performance.now()` exists in jsdom. `Image` does too, but it never fires
// `load`, which is harmless here: the preloader only sets `src`.
