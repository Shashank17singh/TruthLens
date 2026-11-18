import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TextResult } from '../components/TextResult';
import type { TextDetails } from '../api/types';

describe('TextResult Component', () => {
  const mockDetails: TextDetails = {
    word_count: 42,
    sentences: [
      {
        start: 0,
        end: 15,
        text: 'This is the first sentence analyzed for stylometrics.',
        probability: 0.22,
      },
      {
        start: 16,
        end: 35,
        text: 'Consequently, the generative AI language model completed this excerpt.',
        probability: 0.94,
      },
      {
        start: 36,
        end: 42,
        text: 'Human editors subsequently reviewed the article.',
        probability: 0.48,
      },
    ],
  };

  it('renders word count, sentence count and probability extremes', () => {
    render(<TextResult details={mockDetails} />);

    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument(); // 3 sentences
    expect(screen.getByText('94%')).toBeInTheDocument(); // max AI
    expect(screen.getByText('22%')).toBeInTheDocument(); // min AI
  });

  it('renders individual sentences with interactive click inspection', () => {
    render(<TextResult details={mockDetails} />);

    const sentenceEl = screen.getByText(
      /Consequently, the generative AI language model completed this excerpt\./
    );
    expect(sentenceEl).toBeInTheDocument();

    // Click on the sentence to inspect
    fireEvent.click(sentenceEl);

    expect(screen.getByText('Inspected Sentence')).toBeInTheDocument();
    expect(screen.getByText('94% AI Probability')).toBeInTheDocument();
  });

  it('falls back cleanly when no sentences are provided', () => {
    render(<TextResult rawText="Fallback raw plain text." />);

    expect(screen.getByText('Fallback raw plain text.')).toBeInTheDocument();
  });
});
