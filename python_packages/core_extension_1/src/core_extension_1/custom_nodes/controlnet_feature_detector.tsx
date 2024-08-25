import React, { useState, ChangeEvent } from 'react';
import { NodeProps } from 'reactflow';

enum FeatureType {
  Canny = 'canny',
  OpenPose = 'openpose'
}

interface ControlNetFeatureDetectorProps extends NodeProps {}

const ControlNetFeatureDetector: React.FC<ControlNetFeatureDetectorProps> = ({ isConnectable }) => {
  const [featureType, setFeatureType] = useState<FeatureType>(FeatureType.Canny);
  const [threshold1, setThreshold1] = useState<number | undefined>(undefined);
  const [threshold2, setThreshold2] = useState<number | undefined>(undefined);

  const handleFeatureTypeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const newFeatureType = e.target.value as FeatureType;
    setFeatureType(newFeatureType);
    if (newFeatureType !== FeatureType.Canny) {
      setThreshold1(undefined);
      setThreshold2(undefined);
    } else {
      setThreshold1(100);
      setThreshold2(200);
    }
  };

  return (
    <div className="node-container">
      <div className="node-title">ControlNet Feature Detector</div>
      <div className="node-content">
        <select
          value={featureType}
          onChange={handleFeatureTypeChange}
        >
          <option value={FeatureType.Canny}>Canny Edge Detection</option>
          <option value={FeatureType.OpenPose}>OpenPose</option>
        </select>
        {featureType === FeatureType.Canny && (
          <>
            <div>
              <label htmlFor="threshold1">Threshold 1</label>
              <input
                id="threshold1"
                type="number"
                value={threshold1 ?? ''}
                onChange={(e) => {
                  const value = e.target.value ? parseInt(e.target.value) : undefined;
                  setThreshold1(value);
                }}
                min={0}
                max={255}
              />
            </div>
            <div>
              <label htmlFor="threshold2">Threshold 2</label>
              <input
                id="threshold2"
                type="number"
                value={threshold2 ?? ''}
                onChange={(e) => {
                  const value = e.target.value ? parseInt(e.target.value) : undefined;
                  setThreshold2(value);
                }}
                min={0}
                max={255}
              />
            </div>
          </>
        )}
      </div>
      <div className="node-input">
        <div className="input-label">Image</div>
        <div className="input-port" />
      </div>
      <div className="node-output">
        <div className="output-label">Control Image</div>
        <div className="output-port" />
      </div>
    </div>
  );
};

export default ControlNetFeatureDetector;
