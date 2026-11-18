import React from 'react';
import { BrowserRouter, Routes, Route, Navigate, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { AuthModal } from './components/AuthModal';
import { LandingPage } from './pages/LandingPage';
import { AnalyzePage } from './pages/AnalyzePage';
import { ResultPage } from './pages/ResultPage';
import { HistoryPage } from './pages/HistoryPage';
import { ApiKeysPage } from './pages/ApiKeysPage';
import { AdminPage } from './pages/AdminPage';
import { VerifyPage } from './pages/VerifyPage';
import { ShieldCheck, Cpu, Lock } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <BrowserRouter>
            <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
              <Navbar />

              <main className="flex-1 pb-16">
                <Routes>
                  <Route path="/" element={<LandingPage />} />
                  <Route path="/analyze" element={<AnalyzePage />} />
                  <Route path="/results/:id" element={<ResultPage />} />
                  <Route path="/history" element={<HistoryPage />} />
                  <Route path="/keys" element={<ApiKeysPage />} />
                  <Route path="/admin" element={<AdminPage />} />
                  <Route path="/verify" element={<VerifyPage />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </main>

              {/* Footer */}
              <footer className="border-t border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/60 backdrop-blur text-xs text-slate-500 py-6">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-brand-600" />
                    <span className="font-bold text-slate-700 dark:text-slate-300">
                      TruthLens
                    </span>
                    <span>— Open Forensic Multi-Modal Detection System</span>
                  </div>

                  <div className="flex items-center gap-4 text-[11px]">
                    <span className="flex items-center gap-1 text-slate-400">
                      <Cpu className="w-3.5 h-3.5" /> CPU-Optimized ONNX
                    </span>
                    <span className="flex items-center gap-1 text-slate-400">
                      <Lock className="w-3.5 h-3.5" /> 24h Privacy Purge
                    </span>
                    <Link to="/verify" className="hover:text-brand-600 dark:hover:text-brand-400">
                      HMAC Verification
                    </Link>
                  </div>
                </div>
              </footer>

              {/* Global Auth Modal */}
              <AuthModal />
            </div>
          </BrowserRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
