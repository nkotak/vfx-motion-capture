/**
 * WebSocket hook for real-time updates
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { JobProgress } from '@/services/api';

interface WebSocketMessage {
  type: 'connected' | 'progress' | 'complete' | 'error' | 'keepalive' | 'result';
  data?: Record<string, unknown>;
  message?: string;
}

interface UseWebSocketOptions {
  onProgress?: (progress: JobProgress) => void;
  onComplete?: (data: Record<string, unknown>) => void;
  onError?: (error: string) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

export function useJobWebSocket(jobId: string | null, options: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const {
    onProgress,
    onComplete,
    onError,
    onConnect,
    onDisconnect,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  const connect = useCallback(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/jobs/${jobId}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        onConnect?.();
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          switch (message.type) {
            case 'connected':
              // Initial connection confirmation
              break;

            case 'progress':
              if (message.data) {
                onProgress?.(message.data as unknown as JobProgress);
              }
              break;

            case 'complete':
              if (message.data) {
                onComplete?.(message.data);
              }
              break;

            case 'error':
              onError?.(message.message || 'Unknown error');
              break;

            case 'keepalive':
              // Server keepalive, no action needed
              break;
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        onDisconnect?.();

        if (autoReconnect && jobId) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.('WebSocket connection error');
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  }, [jobId, onProgress, onComplete, onError, onConnect, onDisconnect, autoReconnect, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
  }, []);

  const sendPing = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send('ping');
    }
  }, []);

  useEffect(() => {
    if (jobId) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [jobId, connect, disconnect]);

  // Send periodic pings to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(sendPing, 25000);
    return () => clearInterval(pingInterval);
  }, [isConnected, sendPing]);

  return {
    isConnected,
    connect,
    disconnect,
    sendPing,
  };
}

export function useRealtimeWebSocket(
  sessionId: string | null,
  options: {
    onFrame?: (frameData: string, latency: number) => void;
    onError?: (error: string) => void;
  } = {}
) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);

  const frameCountRef = useRef(0);
  const lastSecondRef = useRef(Date.now());

  const { onFrame, onError } = options;

  const connect = useCallback(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime/${sessionId}`;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        frameCountRef.current = 0;
        lastSecondRef.current = Date.now();
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type === 'result') {
            onFrame?.(message.data, message.latency_ms);
            setLatency(message.latency_ms);

            // Calculate FPS
            frameCountRef.current++;
            const now = Date.now();
            if (now - lastSecondRef.current >= 1000) {
              setFps(frameCountRef.current);
              frameCountRef.current = 0;
              lastSecondRef.current = now;
            }
          } else if (message.type === 'error') {
            onError?.(message.message);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      ws.onerror = () => {
        onError?.('WebSocket error');
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  }, [sessionId, onFrame, onError]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendFrame = useCallback((frameData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: frameData,
        timestamp: Date.now(),
      }));
    }
  }, []);

  useEffect(() => {
    if (sessionId) {
      connect();
    } else {
      disconnect();
    }

    return () => {
      disconnect();
    };
  }, [sessionId, connect, disconnect]);

  return {
    isConnected,
    fps,
    latency,
    sendFrame,
    connect,
    disconnect,
  };
}
