export async function GET() {
  try {
    // Instead of SSE, make a simple HTTP request to get recent logs
    const response = await fetch('http://5.78.71.231:8080/api/logs/recent?limit=50&type=all', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch logs: ${response.status}`);
    }

    const vpsData = await response.json();
    
    // Extract logs from VPS response and pass them through
    return Response.json({
      success: true,
      logs: vpsData.logs || [],
      count: vpsData.count || 0,
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