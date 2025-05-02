// Web worker for file processing
self.onmessage = async (e) => {
  const file = e.data;
  
  try {
    // Process the file here
    const result = {
      name: file.name,
      size: file.size,
      type: file.type,
      // Add any additional processing here
    };
    
    self.postMessage({ type: 'success', data: result });
  } catch (error: unknown) {
    self.postMessage({ 
      type: 'error', 
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    });
  }
};

export {}; 