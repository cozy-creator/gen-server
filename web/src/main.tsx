import React from 'react';
import * as ReactDOM from 'react-dom/client';
// import './index.css';
import GraphEditor from '@cozy-creator/graph-editor';
import "@cozy-creator/graph-editor/dist/css";

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <GraphEditor />
  </React.StrictMode>
);
