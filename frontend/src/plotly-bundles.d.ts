// The partial Plotly builds share the full build's API and types.
declare module 'plotly.js-basic-dist-min' {
  import * as Plotly from 'plotly.js'
  export = Plotly
}

declare module 'plotly.js-gl2d-dist-min' {
  import * as Plotly from 'plotly.js'
  export = Plotly
}
