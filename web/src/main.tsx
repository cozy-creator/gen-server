import React from 'react';
import * as ReactDOM from 'react-dom/client';
// import './index.css';
import GraphEditor from '@comfy-creator/graph-editor';
import "@comfy-creator/graph-editor/css";

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <GraphEditor />
  </React.StrictMode>
);
