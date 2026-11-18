import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { api, type AuthUser, getStoredToken } from '../api/client';

interface AuthContextType {
  user: AuthUser | null;
  loading: boolean;
  login: (email: string, pass: string) => Promise<void>;
  register: (email: string, pass: string, name?: string) => Promise<void>;
  loginAsDemo: () => Promise<void>;
  logout: () => void;
  isAuthModalOpen: boolean;
  openAuthModal: () => void;
  closeAuthModal: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  const fetchUser = useCallback(async () => {
    if (!getStoredToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const u = await api.getCurrentUser();
      setUser(u);
    } catch {
      setUser(null);
      api.logout();
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  const login = async (email: string, pass: string) => {
    await api.login(email, pass);
    await fetchUser();
    setIsAuthModalOpen(false);
  };

  const register = async (email: string, pass: string, name?: string) => {
    await api.register(email, pass, name);
    await api.login(email, pass);
    await fetchUser();
    setIsAuthModalOpen(false);
  };

  const loginAsDemo = async () => {
    const demoEmail = 'analyst@truthlens.ai';
    const demoPass = 'TruthLens2026!';
    try {
      await api.login(demoEmail, demoPass);
    } catch {
      // If demo account doesn't exist, register then login
      try {
        await api.register(demoEmail, demoPass, 'Forensic Analyst');
        await api.login(demoEmail, demoPass);
      } catch {
        // Fallback random guest
        const rand = Math.floor(Math.random() * 100000);
        const guestEmail = `guest_${rand}@truthlens.ai`;
        await api.register(guestEmail, 'TruthLens2026!', `Guest ${rand}`);
        await api.login(guestEmail, 'TruthLens2026!');
      }
    }
    await fetchUser();
    setIsAuthModalOpen(false);
  };

  const logout = () => {
    api.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        loginAsDemo,
        logout,
        isAuthModalOpen,
        openAuthModal: () => setIsAuthModalOpen(true),
        closeAuthModal: () => setIsAuthModalOpen(false),
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
