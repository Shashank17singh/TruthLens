export type Modality = 'image' | 'audio' | 'text' | 'url';
export type AnalysisStatus = 'queued' | 'processing' | 'completed' | 'failed';
export type Verdict = 'likely_real' | 'uncertain' | 'likely_fake';

export interface ConfidenceBand {
  low: number;
  high: number;
  level?: number;
}

export interface ImageRegion {
  box: [number, number, number, number]; // [left, top, right, bottom]
  score: number;
  part: string;
}

export interface AudioWindow {
  start: number;
  end: number;
  probability: number;
}

export interface TextSentence {
  start: number;
  end: number;
  text: string;
  probability: number;
}

export interface ImageDetails {
  face_detected: boolean;
  regions: ImageRegion[];
  artefacts: {
    heatmap?: string;
    spectrum?: string;
  };
}

export interface AudioDetails {
  duration_seconds: number;
  aggregate: {
    mean: number;
    max: number;
  };
  windows: AudioWindow[];
  artefacts: {
    saliency?: string;
  };
}

export interface TextDetails {
  word_count: number;
  sentences: TextSentence[];
  artefacts?: Record<string, string>;
}

export interface AnalysisResult {
  probability: number;
  confidence_band: ConfidenceBand;
  verdict: Verdict;
  verdict_label: string;
  explanation: string;
  model_name: string;
  model_version: string;
  warnings: string[];
  disclaimer: string;
  details: ImageDetails | AudioDetails | TextDetails | Record<string, unknown>;
  latency_ms?: number;
}

export interface AnalysisError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface Analysis {
  id: string;
  modality: 'image' | 'audio' | 'text' | null;
  status: AnalysisStatus;
  progress: number;
  created_at: string;
  completed_at?: string | null;
  media_expires_at?: string | null;
  retain_media: boolean;
  file_name?: string | null;
  result?: AnalysisResult | null;
  error?: AnalysisError | null;
}

export interface PaginatedAnalyses {
  items: Analysis[];
  page: number;
  page_size: number;
  total: number;
}

export interface SubmissionResponse {
  analysis_id: string;
  status: string;
  status_url: string;
}

export interface ReportStatusResponse {
  report_id: string;
  status: 'queued' | 'rendering' | 'ready' | 'failed' | 'expired';
  pdf_url: string;
  public_url: string;
}

export interface VerifyResponse {
  genuine: boolean;
  report_id?: string;
  issued_at?: string;
  sha256?: string;
  message?: string;
}

export interface ApiKeyItem {
  id: string;
  name: string;
  key_prefix: string;
  rate_limit_per_min: number;
  last_used_at?: string | null;
  expires_at?: string | null;
  created_at: string;
}

export interface NewApiKeyResponse extends ApiKeyItem {
  key: string;
}

export interface FeedbackSubmission {
  is_correct: boolean;
  reported_label?: 'real' | 'fake' | 'unsure';
  comment?: string;
}

export interface AdminStats {
  total_analyses: number;
  completed_analyses: number;
  modality_breakdown: Record<string, number>;
  total_feedback: number;
  feedback_accuracy: number;
  average_latency_ms: number;
}

export interface FeedbackItem {
  id: number;
  analysis_id: string;
  user_id: string;
  is_correct: boolean;
  reported_label?: string;
  comment?: string;
  created_at: string;
}

export interface PaginatedFeedback {
  items: FeedbackItem[];
  page: number;
  page_size: number;
  total: number;
}

export interface ModelCatalogEntry {
  modality: string;
  name: string;
  version: string;
  decision_threshold: number;
}
