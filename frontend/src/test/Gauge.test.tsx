import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Gauge } from '../components/Gauge';

describe('Gauge Component', () => {
  it('renders probability percentage and label correctly', () => {
    render(
      <Gauge
        probability={0.85}
        confidenceBand={{ low: 0.78, high: 0.91 }}
        verdict="likely_fake"
        verdictLabel="Likely AI-Generated"
      />
    );

    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('Likely AI-Generated')).toBeInTheDocument();
    expect(screen.getByText('[78%, 91%]')).toBeInTheDocument();
  });

  it('exposes accessible ARIA meter attributes', () => {
    render(
      <Gauge
        probability={0.25}
        confidenceBand={{ low: 0.18, high: 0.32 }}
        verdict="likely_real"
        verdictLabel="Likely Real"
      />
    );

    const meter = screen.getByRole('meter');
    expect(meter).toBeInTheDocument();
    expect(meter).toHaveAttribute('aria-valuenow', '25');
    expect(meter).toHaveAttribute('aria-valuemin', '0');
    expect(meter).toHaveAttribute('aria-valuemax', '100');
    expect(meter).toHaveAttribute(
      'aria-valuetext',
      'Likely Real: 25% synthetic probability, 95% confidence interval [18%, 32%]'
    );
  });

  it('handles uncertain verdict and default confidence band', () => {
    render(
      <Gauge
        probability={0.52}
        verdict="uncertain"
        verdictLabel="Uncertain / Mixed"
      />
    );

    expect(screen.getByText('52%')).toBeInTheDocument();
    expect(screen.getByText('Uncertain / Mixed')).toBeInTheDocument();
  });
});
