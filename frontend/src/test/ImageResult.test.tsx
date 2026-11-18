import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ImageResult } from '../components/ImageResult';
import type { ImageDetails } from '../api/types';

describe('ImageResult Component', () => {
  const mockDetails: ImageDetails = {
    face_detected: true,
    regions: [
      {
        box: [0.2, 0.2, 0.8, 0.8],
        score: 0.89,
        part: 'focal_face',
      },
    ],
    artefacts: {
      heatmap: 'heatmap.png',
      spectrum: 'spectrum.png',
    },
  };

  it('renders tab buttons and controls', () => {
    render(<ImageResult analysisId="test-id" details={mockDetails} />);

    expect(screen.getByText('Heatmap Overlay')).toBeInTheDocument();
    expect(screen.getByText('Side-by-Side')).toBeInTheDocument();
    expect(screen.getByText('Frequency Spectrum')).toBeInTheDocument();
  });

  it('allows switching to side-by-side view', () => {
    render(<ImageResult analysisId="test-id" details={mockDetails} />);

    const sideBySideBtn = screen.getByText('Side-by-Side');
    fireEvent.click(sideBySideBtn);

    expect(screen.getByText('Original Media')).toBeInTheDocument();
    expect(screen.getByText('Grad-CAM Attribution Map')).toBeInTheDocument();
  });

  it('allows switching to frequency spectrum tab', () => {
    render(<ImageResult analysisId="test-id" details={mockDetails} />);

    const spectrumBtn = screen.getByText('Frequency Spectrum');
    fireEvent.click(spectrumBtn);

    expect(
      screen.getByText('2D Discrete Cosine Transform (DCT) Frequency Domain')
    ).toBeInTheDocument();
  });

  it('renders bounding boxes when toggled', () => {
    render(<ImageResult analysisId="test-id" details={mockDetails} />);

    expect(screen.getByText(/focal_face \(89%\)/)).toBeInTheDocument();
  });
});
