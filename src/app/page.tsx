'use client';

import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';

interface DataPreview {
  columns: string[];
  data: Record<string, any>[];
}

const PROGRESS_STEPS = [
  { label: 'Uploading Data', key: 'upload' },
  { label: 'Getting AI Insights', key: 'ai' },
  { label: 'Creating Report', key: 'report' },
  { label: 'Done!', key: 'done' }
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null);
  const [visualizations, setVisualizations] = useState<any[]>([]);
  const [showJson, setShowJson] = useState(false);
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState(PROGRESS_STEPS[0].label);

  // Only log when dataPreview actually changes to a value
  useEffect(() => {
    if (dataPreview) {
      console.log('Data preview updated:', {
        columns: dataPreview.columns,
        rowCount: dataPreview.data.length
      });
    }
  }, [dataPreview]);

  // Function to convert column index to Excel-style letter
  const getColumnLetter = (index: number): string => {
    let letter = '';
    while (index >= 0) {
      letter = String.fromCharCode(65 + (index % 26)) + letter;
      index = Math.floor(index / 26) - 1;
    }
    return letter;
  };

  // Helper to simulate progress
  const nextProgress = (step: number) => {
    setProgress(step);
    setProgressLabel(PROGRESS_STEPS[step].label);
  };

  const handleUpload = useCallback(async (file: File) => {
    try {
      console.log('📤 Uploading file:', file.name);
      setIsProcessing(true);
      setError(null);
      
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      console.log('📡 Server response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ Server error response:', errorData);
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      console.log('📦 Server response:', data);

      if (!data.preview || !data.preview.columns || !data.preview.data) {
        console.error('❌ Invalid preview data structure:', data);
        throw new Error('Invalid preview data structure');
      }

      setDataPreview({
        columns: data.preview.columns,
        data: data.preview.data
      });
      console.log('✅ Data preview updated:', {
        columnCount: data.preview.columns.length,
        rowCount: data.preview.data.length
      });

    } catch (error) {
      console.error('❌ Upload error:', error);
      setError(error instanceof Error ? error.message : 'Failed to upload file');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      console.log('📄 File selected:', file.name);
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        console.log('❌ File too large');
        setError('File size must be less than 10MB');
        return;
      }
      // Validate file type
      if (!file.type.match(/application\/vnd.openxmlformats-officedocument.spreadsheetml.sheet|text\/csv/)) {
        console.log('❌ Invalid file type');
        setError('Please upload an Excel (.xlsx) or CSV file');
        return;
      }

      setFile(file);
      setError(null);

      // For Excel files, trigger upload immediately
      if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
        console.log('📊 Processing Excel file');
        handleUpload(file);
        return;
      }

      // For CSV files, read and preview immediately
      console.log('📊 Processing CSV file');
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const content = e.target?.result as string;
          let previewData: Record<string, string>[];
          let columns: string[];

          // Parse CSV
          const lines = content.split('\n');
          columns = lines[0].split(',').map(col => col.trim());
          
          previewData = lines.slice(1, 11).map(line => {
            const values = line.split(',');
            return columns.reduce((obj: Record<string, string>, col: string, i: number) => {
              obj[col] = values[i]?.trim() || '';
              return obj;
            }, {} as Record<string, string>);
          });

          setDataPreview({
            columns,
            data: previewData
          });
        } catch (error) {
          console.error('❌ CSV parsing error:', error);
          setError('Error previewing file. Please try uploading.');
        }
      };

      reader.readAsText(file);
    }
  }, [handleUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    multiple: false,
    maxSize: 10 * 1024 * 1024, // 10MB
    onDragEnter: (e) => {
      e.preventDefault();
      e.stopPropagation();
    },
    onDragOver: (e) => {
      e.preventDefault();
      e.stopPropagation();
    },
    onDragLeave: (e) => {
      e.preventDefault();
      e.stopPropagation();
    }
  });

  const handleGenerateDashboard = async () => {
    if (!file) return;

    setIsProcessing(true);
    setError(null);  // Clear any previous errors
    nextProgress(0); // Uploading Data
    // Read the file as base64 (for progress simulation only)
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        // No need to store in localStorage, just pass file via router state
        nextProgress(1); // Getting AI Insights
        // Convert file to Blob (already a File object)
        // Send to backend
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate-dashboard`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to generate dashboard');
        }
        nextProgress(2); // Creating Report
        const data = await response.json();
        // Pass the visualizations via router state (use query string for Next.js)
        const vizParam = encodeURIComponent(btoa(JSON.stringify(data.visualizations)));
        nextProgress(3); // Done!
        setTimeout(() => {
          setIsProcessing(false);
          router.push(`/visualizations?data=${vizParam}`);
        }, 500); // Short delay for smoothness
      } catch (err: any) {
        setError(err.message || 'Failed to generate report');
        setIsProcessing(false);
      }
    };
    reader.readAsDataURL(file); // Still triggers progress, but not used for storage
  };

  const handleViewVisualizations = () => {
    if (visualizations.length > 0) {
      const encodedData = encodeURIComponent(JSON.stringify(visualizations));
      router.push(`/visualizations?data=${encodedData}`);
    }
  };

  // Progress bar component
  const ProgressBar = ({ step }: { step: number }) => (
    <div className="w-full max-w-xl mx-auto mb-8">
      <div className="flex justify-between mb-2">
        {PROGRESS_STEPS.map((s, i) => (
          <div key={s.key} className={`text-xs font-semibold ${i <= step ? 'text-blue-700' : 'text-gray-400'}`}>{s.label}</div>
        ))}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div className="bg-blue-600 h-2 rounded-full transition-all duration-500" style={{ width: `${(step / (PROGRESS_STEPS.length - 1)) * 100}%` }}></div>
      </div>
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Turn Excel into Insights
        </h1>
        <p className="text-xl text-gray-600">
          Upload your Excel file and get beautiful, interactive visualizations with AI-generated insights in seconds.
        </p>
      </div>

      <div className="max-w-4xl mx-auto">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors relative
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
            ${error ? 'border-red-500 bg-red-50' : ''}`}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <div className="absolute inset-0 bg-blue-50 bg-opacity-90 flex items-center justify-center rounded-lg">
              <p className="text-blue-500 text-lg font-medium">Drop your file here...</p>
            </div>
          ) : (
            <div>
              <p className="text-gray-600 mb-2">Drag and drop your Excel file here, or click to select</p>
              <p className="text-sm text-gray-500">Supports .xlsx and .csv files (max 10MB)</p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg">
            {error}
          </div>
        )}

        {file && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-gray-700">Selected file: {file.name}</p>
            <p className="text-sm text-gray-500">Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
            {!isProcessing && (
              <div className="flex gap-4 mb-8">
                <button
                  onClick={handleGenerateDashboard}
                  className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
                >
                  {isProcessing ? 'Generating...' : 'Generate Report'}
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading Indicator */}
        {isProcessing && (
          <div className="mt-8 p-8 border-2 border-gray-200 rounded-lg bg-white">
            <ProgressBar step={progress} />
            <div className="flex flex-col items-center justify-center space-y-4">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent"></div>
              <div className="text-center">
                <p className="text-lg font-medium text-gray-900">{progressLabel}</p>
                <p className="text-sm text-gray-500 mt-1">Generating your report, please wait...</p>
              </div>
            </div>
          </div>
        )}

        {/* Preview Table */}
        {dataPreview && dataPreview.columns && dataPreview.data && (
          <div className="mt-8">
            <h3 className="text-lg font-semibold mb-4">Data Preview</h3>
            <div className="overflow-x-auto border border-gray-200 rounded-lg shadow">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    {/* Row number header */}
                    <th className="w-12 px-3 py-2 text-left text-xs font-medium text-gray-500 bg-gray-100 border-r border-gray-200"></th>
                    {/* Column letters */}
                    {dataPreview.columns.map((_, index) => (
                      <th
                        key={`col-${index}`}
                        className="px-6 py-2 text-center text-xs font-medium text-gray-500 border-r border-gray-200 bg-gray-100"
                      >
                        {getColumnLetter(index)}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {/* Row number header */}
                    <th className="w-12 px-3 py-2 text-left text-xs font-medium text-gray-500 bg-gray-100 border-r border-gray-200"></th>
                    {/* Column names */}
                    {dataPreview.columns.map((column, index) => (
                      <th
                        key={index}
                        className="px-6 py-2 text-left text-xs font-medium text-gray-500 border-r border-gray-200"
                      >
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {dataPreview.data.map((row, rowIndex) => (
                    <tr key={rowIndex} className="hover:bg-gray-50">
                      {/* Row number */}
                      <td className="w-12 px-3 py-2 text-xs text-gray-500 bg-gray-100 border-r border-gray-200 font-medium">
                        {rowIndex + 1}
                      </td>
                      {/* Row data */}
                      {dataPreview.columns.map((column, colIndex) => (
                        <td
                          key={`${rowIndex}-${colIndex}`}
                          className="px-6 py-2 text-sm text-gray-500 border-r border-gray-200 whitespace-nowrap"
                        >
                          {row[column]?.toString() || ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 