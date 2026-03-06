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
    onFrame?: (frameUrl: string, latency: number) => void;
    onError?: (error: string) => void;
  } = {}
) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);

  const frameCountRef = useRef(0);
  const lastSecondRef = useRef(Date.now());
  const frameInFlightRef = useRef(false);
  const lastFrameSentAtRef = useRef<number | null>(null);

  const { onFrame, onError } = options;

  const handleFrameResult = useCallback((frameUrl: string, reportedLatency?: number) => {
    const measuredLatency = lastFrameSentAtRef.current !== null
      ? performance.now() - lastFrameSentAtRef.current
      : 0;
    const nextLatency = reportedLatency ?? measuredLatency;

    frameInFlightRef.current = false;
    lastFrameSentAtRef.current = null;

    onFrame?.(frameUrl, nextLatency);
    setLatency(nextLatency);

    frameCountRef.current++;
    const now = Date.now();
    if (now - lastSecondRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastSecondRef.current = now;
    }
  }, [onFrame]);

  const canSendFrame = useCallback(() => (
    wsRef.current?.readyState === WebSocket.OPEN
    && !frameInFlightRef.current
    && wsRef.current.bufferedAmount < 1_000_000
  ), []);

  const connect = useCallback(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/realtime/${sessionId}`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'blob';

      ws.onopen = () => {
        setIsConnected(true);
        frameCountRef.current = 0;
        lastSecondRef.current = Date.now();
        frameInFlightRef.current = false;
        lastFrameSentAtRef.current = null;
      };

      ws.onmessage = (event) => {
        try {
          if (typeof event.data === 'string') {
            const message = JSON.parse(event.data);

            if (message.type === 'result' && typeof message.data === 'string') {
              handleFrameResult(
                `data:image/jpeg;base64,${message.data}`,
                typeof message.latency_ms === 'number' ? message.latency_ms : undefined
              );
            } else if (message.type === 'error') {
              frameInFlightRef.current = false;
              lastFrameSentAtRef.current = null;
              onError?.(message.message);
            }
            return;
          }

          if (event.data instanceof Blob) {
            const frameUrl = URL.createObjectURL(event.data);
            if (onFrame) {
              handleFrameResult(frameUrl);
            } else {
              URL.revokeObjectURL(frameUrl);
              frameInFlightRef.current = false;
              lastFrameSentAtRef.current = null;
            }
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
          frameInFlightRef.current = false;
          lastFrameSentAtRef.current = null;
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        frameInFlightRef.current = false;
        lastFrameSentAtRef.current = null;
      };

      ws.onerror = () => {
        frameInFlightRef.current = false;
        lastFrameSentAtRef.current = null;
        onError?.('WebSocket error');
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
    }
  }, [sessionId, onFrame, onError]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'stop' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    frameInFlightRef.current = false;
    lastFrameSentAtRef.current = null;
    setIsConnected(false);
  }, []);

  const sendFrame = useCallback((frameData: Blob) => {
    if (!canSendFrame() || !wsRef.current) {
      return false;
    }

    frameInFlightRef.current = true;
    lastFrameSentAtRef.current = performance.now();
    wsRef.current.send(frameData);
    return true;
  }, [canSendFrame]);

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
    canSendFrame,
    connect,
    disconnect,
  };
}
