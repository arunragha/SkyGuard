import { useEffect, useRef, useState, useCallback } from "react";
import { Incident, StationMeta } from "../types";

export interface SSEState {
  connected: boolean;
  reconnecting: boolean;
  lastUpdatedSecondsAgo: number;
  lastHeartbeatTime: Date | null;
}

export function useSSE(
  onIncidentReceived: (incident: Incident) => void,
  onStationHealthUpdate: (stationId: string, health: string) => void,
  onReadingReceived?: (reading: any) => void
) {
  const [sseState, setSseState] = useState<SSEState>({
    connected: false,
    reconnecting: false,
    lastUpdatedSecondsAgo: 0,
    lastHeartbeatTime: null,
  });

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);
  const backoffDelayRef = useRef<number>(1000); // 1s initial backoff
  const lastEventTimeRef = useRef<number>(Date.now());

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const es = new EventSource("/stream");
      eventSourceRef.current = es;

      es.onopen = () => {
        backoffDelayRef.current = 1000;
        lastEventTimeRef.current = Date.now();
        setSseState((prev) => ({
          ...prev,
          connected: true,
          reconnecting: false,
        }));
      };

      es.addEventListener("heartbeat", (e: MessageEvent) => {
        lastEventTimeRef.current = Date.now();
        try {
          const data = JSON.parse(e.data);
          setSseState((prev) => ({
            ...prev,
            connected: true,
            reconnecting: false,
            lastHeartbeatTime: new Date(data.timestamp || Date.now()),
          }));
        } catch (err) {}
      });

      es.addEventListener("reading", (e: MessageEvent) => {
        lastEventTimeRef.current = Date.now();
        try {
          const data = JSON.parse(e.data);
          if (onReadingReceived) {
            onReadingReceived(data);
          }
        } catch (err) {}
      });

      es.addEventListener("anomaly", (e: MessageEvent) => {
        lastEventTimeRef.current = Date.now();
        try {
          const data = JSON.parse(e.data);
          const incident: Incident = {
            id: `inc_${data.station_id}_${Date.now()}`,
            timestamp: data.timestamp || new Date().toISOString(),
            station_id: data.station_id,
            station_name: data.station_name || data.station_id,
            latitude: data.latitude || 0,
            longitude: data.longitude || 0,
            anomaly_score: data.anomaly_score || 0,
            anomaly_threshold: data.anomaly_threshold || 0.5,
            predicted_type: data.predicted_type || "spike",
            severity_level: data.severity_level || "medium",
            glyph: data.glyph || "•",
            diagnosed_sensor: data.diagnosed_sensor || "unknown",
            contributing_sensors: data.contributing_sensors || [],
            readings: data.readings || {},
            is_new: true,
          };
          onIncidentReceived(incident);
        } catch (err) {}
      });

      es.addEventListener("station_health", (e: MessageEvent) => {
        lastEventTimeRef.current = Date.now();
        try {
          const data = JSON.parse(e.data);
          if (data.station_id && data.health) {
            onStationHealthUpdate(data.station_id, data.health);
          }
        } catch (err) {}
      });

      es.onerror = () => {
        es.close();
        eventSourceRef.current = null;
        setSseState((prev) => ({
          ...prev,
          connected: false,
          reconnecting: true,
        }));

        // Exponential backoff reconnect: 1s, 2s, 4s, 8s, max 15s
        const nextDelay = Math.min(15000, backoffDelayRef.current * 1.8);
        backoffDelayRef.current = nextDelay;
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = setTimeout(connect, nextDelay);
      };
    } catch (err) {
      setSseState((prev) => ({
        ...prev,
        connected: false,
        reconnecting: true,
      }));
    }
  }, [onIncidentReceived, onStationHealthUpdate, onReadingReceived]);

  useEffect(() => {
    connect();

    // Heartbeat ticker updating seconds ago every 1s
    const tickerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - lastEventTimeRef.current) / 1000);
      setSseState((prev) => ({
        ...prev,
        lastUpdatedSecondsAgo: elapsedSec,
      }));
    }, 1000);

    return () => {
      clearInterval(tickerInterval);
      clearTimeout(reconnectTimeoutRef.current);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [connect]);

  return sseState;
}
