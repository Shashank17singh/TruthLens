import React, { useRef, useState } from 'react';
import { AlertCircle, X, Image as ImageIcon, Music } from 'lucide-react';

interface DropzoneProps {
  acceptType: 'image' | 'audio';
  maxSizeMB?: number;
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  acceptType,
  maxSizeMB = acceptType === 'image' ? 10 : 25,
  onFileSelect,
  selectedFile,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const acceptedMimeTypes =
    acceptType === 'image'
      ? ['image/jpeg', 'image/png', 'image/webp']
      : ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-flac'];

  const acceptedExtensions = acceptType === 'image' ? '.jpg,.jpeg,.png,.webp' : '.wav,.mp3,.flac';

  const validateAndSetFile = (file: File) => {
    setErrorMessage(null);

    // Size check
    const maxBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxBytes) {
      setErrorMessage(`File exceeds the ${maxSizeMB} MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`);
      return;
    }

    // MIME / Extension check
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    const isMimeValid = acceptedMimeTypes.includes(file.type.toLowerCase());
    const isExtValid = acceptedExtensions.split(',').includes(extension);

    if (!isMimeValid && !isExtValid) {
      setErrorMessage(
        `Invalid file type. Allowed: ${acceptedExtensions.toUpperCase().replace(/\./g, ' ')}.`
      );
      return;
    }

    onFileSelect(file);

    // Create thumbnail if image
    if (acceptType === 'image') {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setErrorMessage(null);
    onFileSelect(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="w-full space-y-3">
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && inputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 ${
            isDragOver
              ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-950/30 scale-[1.01]'
              : 'border-slate-300 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-900/50 hover:bg-slate-100/60 dark:hover:bg-slate-800/40'
          } ${disabled ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept={acceptedExtensions}
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                validateAndSetFile(e.target.files[0]);
              }
            }}
            disabled={disabled}
          />

          <div className="w-12 h-12 rounded-2xl bg-white dark:bg-slate-800 shadow-sm border border-slate-200 dark:border-slate-700 flex items-center justify-center text-brand-600 dark:text-brand-400 mb-3">
            {acceptType === 'image' ? <ImageIcon className="w-6 h-6" /> : <Music className="w-6 h-6" />}
          </div>

          <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
            Click to upload or drag & drop {acceptType}
          </p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            {acceptType === 'image' ? 'PNG, JPEG, WebP' : 'WAV, MP3, FLAC'} (Max {maxSizeMB} MB)
          </p>
        </div>
      ) : (
        <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex items-center justify-between gap-4 shadow-sm">
          <div className="flex items-center gap-3 overflow-hidden">
            {acceptType === 'image' && previewUrl ? (
              <img
                src={previewUrl}
                alt="Upload preview"
                className="w-12 h-12 rounded-lg object-cover border border-slate-200 dark:border-slate-700 flex-shrink-0"
              />
            ) : (
              <div className="w-12 h-12 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-600 dark:text-slate-300 flex-shrink-0">
                {acceptType === 'image' ? <ImageIcon className="w-6 h-6" /> : <Music className="w-6 h-6" />}
              </div>
            )}

            <div className="truncate">
              <p className="text-sm font-semibold text-slate-800 dark:text-slate-200 truncate">
                {selectedFile.name}
              </p>
              <p className="text-xs text-slate-500 font-mono">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </p>
            </div>
          </div>

          {!disabled && (
            <button
              onClick={handleClear}
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              title="Remove file"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      )}

      {errorMessage && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-rose-50 dark:bg-rose-950/50 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}
    </div>
  );
};
