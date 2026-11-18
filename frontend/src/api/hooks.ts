import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './client';
import type { FeedbackSubmission } from './types';

export function useCurrentUser() {
  return useQuery({
    queryKey: ['currentUser'],
    queryFn: () => api.getCurrentUser(),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });
}

export function useSubmitImage() {
  return useMutation({
    mutationFn: ({ file, retainMedia }: { file: File; retainMedia?: boolean }) =>
      api.analyzeImage(file, retainMedia),
  });
}

export function useSubmitAudio() {
  return useMutation({
    mutationFn: ({ file, retainMedia }: { file: File; retainMedia?: boolean }) =>
      api.analyzeAudio(file, retainMedia),
  });
}

export function useSubmitText() {
  return useMutation({
    mutationFn: ({ text, retainMedia }: { text: string; retainMedia?: boolean }) =>
      api.analyzeText(text, retainMedia),
  });
}

export function useSubmitUrl() {
  return useMutation({
    mutationFn: ({ url, modalityHint, retainMedia }: { url: string; modalityHint?: string; retainMedia?: boolean }) =>
      api.analyzeUrl(url, modalityHint, retainMedia),
  });
}

export function useAnalysis(id: string | undefined, enabled = true, refetchInterval?: number | false) {
  return useQuery({
    queryKey: ['analysis', id],
    queryFn: () => {
      if (!id) throw new Error('Missing analysis ID');
      return api.getAnalysis(id);
    },
    enabled: Boolean(id) && enabled,
    refetchInterval,
  });
}

export function useAnalyses(params?: {
  page?: number;
  page_size?: number;
  modality?: string;
  status?: string;
  q?: string;
}) {
  return useQuery({
    queryKey: ['analyses', params],
    queryFn: () => api.listAnalyses(params),
  });
}

export function useSaveAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.saveAnalysis(id),
    onSuccess: (data) => {
      queryClient.setQueryData(['analysis', data.id], data);
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
    },
  });
}

export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analyses'] });
    },
  });
}

export function useSubmitFeedback(analysisId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (feedback: FeedbackSubmission) => api.submitFeedback(analysisId, feedback),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis', analysisId] });
      queryClient.invalidateQueries({ queryKey: ['adminFeedback'] });
      queryClient.invalidateQueries({ queryKey: ['adminStats'] });
    },
  });
}

export function useRequestReport() {
  return useMutation({
    mutationFn: ({ analysisId, format }: { analysisId: string; format?: string }) =>
      api.requestReport(analysisId, format),
  });
}

export function useReport(reportId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ['report', reportId],
    queryFn: () => {
      if (!reportId) throw new Error('Missing report ID');
      return api.getReport(reportId);
    },
    enabled: Boolean(reportId) && enabled,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data || data.status === 'queued' || data.status === 'rendering') {
        return 1500;
      }
      return false;
    },
  });
}

export function useVerifyReport() {
  return useMutation({
    mutationFn: ({ file, reportId, sha256 }: { file?: File; reportId?: string; sha256?: string }) =>
      api.verifyReport(file, reportId, sha256),
  });
}

export function useApiKeys() {
  return useQuery({
    queryKey: ['apiKeys'],
    queryFn: () => api.listApiKeys(),
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ name, rateLimit }: { name: string; rateLimit?: number }) =>
      api.createApiKey(name, rateLimit),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['apiKeys'] });
    },
  });
}

export function useRevokeApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.revokeApiKey(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['apiKeys'] });
    },
  });
}

export function useAdminStats() {
  return useQuery({
    queryKey: ['adminStats'],
    queryFn: () => api.getAdminStats(),
  });
}

export function useAdminFeedback(page = 1, pageSize = 20) {
  return useQuery({
    queryKey: ['adminFeedback', page, pageSize],
    queryFn: () => api.listAdminFeedback(page, pageSize),
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => api.listModels(),
    staleTime: 10 * 60 * 1000,
  });
}
