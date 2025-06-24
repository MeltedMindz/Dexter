export async function GET() {
  try {
    // Get the DexBrain API URL from environment variables
    const dexbrainUrl = process.env.DEXBRAIN_API_URL || process.env.NEXT_PUBLIC_DEXBRAIN_API_URL || 'http://localhost:8080';
    
    // Instead of SSE, make a simple HTTP request to get recent logs
    const response = await fetch(`${dexbrainUrl}/api/logs/recent?limit=50&type=all`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch logs: ${response.status}`);
    }

    const logsData = await response.json();
    
    // Extract logs from DexBrain API response and pass them through
    return Response.json({
      success: true,
      logs: logsData.logs || [],
      count: logsData.count || 0,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching logs:', error);
    
    // Return error without mock data - will show connection error in BrainWindow
    return Response.json({
      success: false,
      logs: [],
      error: `Failed to connect to VPS log stream: ${error instanceof Error ? error.message : 'Unknown error'}`,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
}