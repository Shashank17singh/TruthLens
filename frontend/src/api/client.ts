import type {
  Analysis,
  PaginatedAnalyses,
  SubmissionResponse,
  ReportStatusResponse,
  VerifyResponse,
  ApiKeyItem,
  NewApiKeyResponse,
  FeedbackSubmission,
  FeedbackItem,
  AdminStats,
  PaginatedFeedback,
  ModelCatalogEntry,
} from './types';

const API_BASE = '/api/v1';

export class ApiError extends Error {
  code: string;
  status: number;
  details?: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export function getStoredToken(): string | null {
  return localStorage.getItem('tl_token');
}

export function setStoredToken(token: string | null): void {
  if (token) {
    localStorage.setItem('tl_token', token);
  } else {
    localStorage.removeItem('tl_token');
  }
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem('tl_refresh_token');
}

export function setStoredRefreshToken(token: string | null): void {
  if (token) {
    localStorage.setItem('tl_refresh_token', token);
  } else {
    localStorage.removeItem('tl_refresh_token');
  }
}

export function getStoredApiKey(): string | null {
  return localStorage.getItem('tl_api_key');
}

export function setStoredApiKey(key: string | null): void {
  if (key) {
    localStorage.setItem('tl_api_key', key);
  } else {
    localStorage.removeItem('tl_api_key');
  }
}

let isRefreshing = false;
let refreshPromise: Promise<string | null> | null = null;

async function refreshAccessToken(): Promise<string | null> {
  if (isRefreshing && refreshPromise) {
    return refreshPromise;
  }
  isRefreshing = true;
  refreshPromise = (async () => {
    try {
      const storedRefresh = getStoredRefreshToken();
      const res = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ refresh_token: storedRefresh || undefined }),
      });
      if (!res.ok) {
        setStoredToken(null);
        setStoredRefreshToken(null);
        return null;
      }
      const data = await res.json();
      if (data.access_token) {
        setStoredToken(data.access_token);
        if (data.refresh_token) {
          setStoredRefreshToken(data.refresh_token);
        }
        return data.access_token;
      }
      return null;
    } catch {
      return null;
    } finally {
      isRefreshing = false;
      refreshPromise = null;
    }
  })();
  return refreshPromise;
}

interface CustomRequestInit extends RequestInit {
  _retry?: boolean;
}

async function request<T>(
  endpoint: string,
  options: CustomRequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers || {});

  const token = getStoredToken();
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const apiKey = getStoredApiKey();
  if (apiKey && !headers.has('X-API-Key') && !headers.has('Authorization')) {
    headers.set('X-API-Key', apiKey);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    credentials: 'include',
    headers,
  });

  if (response.status === 401 && !options._retry && endpoint !== '/auth/login' && endpoint !== '/auth/refresh') {
    const newToken = await refreshAccessToken();
    if (newToken) {
      return request<T>(endpoint, {
        ...options,
        _retry: true,
      });
    }
  }

  if (!response.ok) {
    let code = 'error';
    let message = `Request failed with status ${response.status}`;
    let details: Record<string, unknown> | undefined;

    try {
      const errData = await response.json();
      if (errData.error) {
        code = errData.error.code || code;
        message = errData.error.message || message;
        details = errData.error.details;
      } else if (errData.detail) {
        message = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
      }
    } catch {
      // Non-JSON response
    }

    throw new ApiError(response.status, code, message, details);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

// Auth API
export interface AuthUser {
  id: string;
  email: string;
  display_name: string | null;
  role: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  token_type: string;
  expires_in: number;
}

export const api = {
  // Authentication
  async register(email: string, password: string, displayName?: string): Promise<AuthUser> {
    return request<AuthUser>('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, display_name: displayName }),
    });
  },

  async login(email: string, password: string): Promise<TokenResponse> {
    const data = await request<TokenResponse>('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    setStoredToken(data.access_token);
    if (data.refresh_token) {
      setStoredRefreshToken(data.refresh_token);
    }
    return data;
  },

  async getCurrentUser(): Promise<AuthUser> {
    return request<AuthUser>('/auth/me');
  },

  logout(): void {
    setStoredToken(null);
    setStoredRefreshToken(null);
  },

  // Analysis Submission
  async analyzeImage(file: File, retainMedia = false): Promise<SubmissionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return request<SubmissionResponse>(`/analyze/image?retain_media=${retainMedia}`, {
      method: 'POST',
      body: formData,
    });
  },

  async analyzeAudio(file: File, retainMedia = false): Promise<SubmissionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return request<SubmissionResponse>(`/analyze/audio?retain_media=${retainMedia}`, {
      method: 'POST',
      body: formData,
    });
  },

  async analyzeText(text: string, retainMedia = false): Promise<SubmissionResponse> {
    return request<SubmissionResponse>(`/analyze/text?retain_media=${retainMedia}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },

  async analyzeUrl(url: string, modalityHint?: string, retainMedia = false): Promise<SubmissionResponse> {
    return request<SubmissionResponse>(`/analyze/url?retain_media=${retainMedia}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, modality_hint: modalityHint }),
    });
  },

  // Analyses Management
  async getAnalysis(id: string): Promise<Analysis> {
    return request<Analysis>(`/analyses/${id}`);
  },

  async listAnalyses(params?: {
    page?: number;
    page_size?: number;
    modality?: string;
    status?: string;
    q?: string;
  }): Promise<PaginatedAnalyses> {
    const query = new URLSearchParams();
    if (params?.page) query.set('page', params.page.toString());
    if (params?.page_size) query.set('page_size', params.page_size.toString());
    if (params?.modality) query.set('modality', params.modality);
    if (params?.status) query.set('status', params.status);
    if (params?.q) query.set('q', params.q);

    const qs = query.toString();
    return request<PaginatedAnalyses>(`/analyses${qs ? `?${qs}` : ''}`);
  },

  async saveAnalysis(id: string): Promise<Analysis> {
    return request<Analysis>(`/analyses/${id}/save`, { method: 'POST' });
  },

  async deleteAnalysis(id: string): Promise<void> {
    await request<void>(`/analyses/${id}`, { method: 'DELETE' });
  },

  async createWebSocketTicket(id: string): Promise<{ ticket: string; expires_in: number }> {
    return request<{ ticket: string; expires_in: number }>(`/analyses/${id}/ws-ticket`, {
      method: 'POST',
    });
  },

  getArtefactUrl(analysisId: string, name: string): string {
    const token = getStoredToken();
    const apiKey = getStoredApiKey();
    let query = '';
    if (token) {
      query = `?token=${encodeURIComponent(token)}`;
    } else if (apiKey) {
      query = `?api_key=${encodeURIComponent(apiKey)}`;
    }
    return `${API_BASE}/analyses/${analysisId}/artefacts/${name}${query}`;
  },

  async getArtefactBlobUrl(analysisId: string, name: string): Promise<string> {
    const token = getStoredToken();
    const apiKey = getStoredApiKey();
    const headers: Record<string, string> = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    } else if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    const query = token ? `?token=${encodeURIComponent(token)}` : (apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : '');
    const res = await fetch(`${API_BASE}/analyses/${analysisId}/artefacts/${name}${query}`, {
      headers,
    });
    if (!res.ok) {
      throw new Error(`Failed to load artefact ${name} (${res.status})`);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },

  // Feedback
  async submitFeedback(id: string, feedback: FeedbackSubmission): Promise<FeedbackItem> {
    return request<FeedbackItem>(`/analyses/${id}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedback),
    });
  },

  // Reports
  async requestReport(analysisId: string, format = 'pdf'): Promise<ReportStatusResponse> {
    return request<ReportStatusResponse>('/reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ analysis_id: analysisId, format }),
    });
  },

  async getReport(reportId: string): Promise<ReportStatusResponse> {
    return request<ReportStatusResponse>(`/reports/${reportId}`);
  },

  async verifyReport(file?: File, reportId?: string, sha256?: string): Promise<VerifyResponse> {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (reportId) {
      formData.append('report_id', reportId);
    }
    if (sha256) {
      formData.append('sha256', sha256);
    }
    return request<VerifyResponse>('/verify', {
      method: 'POST',
      body: formData,
    });
  },

  // API Keys
  async listApiKeys(): Promise<ApiKeyItem[]> {
    return request<ApiKeyItem[]>('/api-keys');
  },

  async createApiKey(name: string, rateLimit = 120): Promise<NewApiKeyResponse> {
    return request<NewApiKeyResponse>('/api-keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, rate_limit_per_min: rateLimit }),
    });
  },

  async revokeApiKey(id: string): Promise<void> {
    await request<void>(`/api-keys/${id}`, { method: 'DELETE' });
  },

  // Admin
  async getAdminStats(): Promise<AdminStats> {
    return request<AdminStats>('/admin/stats');
  },

  async listAdminFeedback(page = 1, pageSize = 20): Promise<PaginatedFeedback> {
    return request<PaginatedFeedback>(`/admin/feedback?page=${page}&page_size=${pageSize}`);
  },

  // Models catalog
  async listModels(): Promise<{ models: ModelCatalogEntry[] }> {
    return request<{ models: ModelCatalogEntry[] }>('/models');
  },
};
