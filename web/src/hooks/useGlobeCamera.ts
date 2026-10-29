import { useRef, useCallback, useEffect } from "react";

export const OVERVIEW = { lat: 20, lng: 0, altitude: 2.2 };
export const OVERVIEW_COORDS = OVERVIEW;
export const FOCUS_ALTITUDE = 0.4;
export const FLY_MS = 1200;

export interface UseGlobeCameraOptions {
  reducedMotion?: boolean;
  onDeselect?: () => void;
}

export function useGlobeCamera(
  globeRef: React.MutableRefObject<any>,
  options: UseGlobeCameraOptions = {}
) {
  const autoRotateResumeTimeoutRef = useRef<any>(null);
  const { reducedMotion = false } = options;

  const getControls = useCallback(() => {
    if (!globeRef.current) return null;
    return typeof globeRef.current.controls === "function"
      ? globeRef.current.controls()
      : globeRef.current.controls;
  }, [globeRef]);

  const setAutoRotate = useCallback(
    (enabled: boolean) => {
      const controls = getControls();
      if (controls) {
        controls.autoRotate = enabled && !reducedMotion;
      }
    },
    [getControls, reducedMotion]
  );

  const flyToIncident = useCallback(
    (lat: number, lng: number) => {
      if (!globeRef.current) return;

      // 1. Pause auto-rotation while an incident is selected
      if (autoRotateResumeTimeoutRef.current) {
        clearTimeout(autoRotateResumeTimeoutRef.current);
        autoRotateResumeTimeoutRef.current = null;
      }
      setAutoRotate(false);

      // 2. Smoothly fly to incident coordinates at altitude 0.4 within ~1.2s
      const duration = reducedMotion ? 0 : FLY_MS;
      globeRef.current.pointOfView({ lat, lng, altitude: FOCUS_ALTITUDE }, duration);
    },
    [globeRef, reducedMotion, setAutoRotate]
  );

  const backToOverview = useCallback(() => {
    if (!globeRef.current) return;

    // 1. Fly camera back to global overview
    const duration = reducedMotion ? 0 : FLY_MS;
    globeRef.current.pointOfView(OVERVIEW, duration);

    // 2. Resume auto-rotation 10 seconds after deselect
    if (autoRotateResumeTimeoutRef.current) {
      clearTimeout(autoRotateResumeTimeoutRef.current);
    }
    autoRotateResumeTimeoutRef.current = setTimeout(() => {
      setAutoRotate(true);
    }, 10000);
  }, [globeRef, reducedMotion, setAutoRotate]);

  useEffect(() => {
    return () => {
      if (autoRotateResumeTimeoutRef.current) {
        clearTimeout(autoRotateResumeTimeoutRef.current);
      }
    };
  }, []);

  return {
    flyToIncident,
    backToOverview,
    setAutoRotate,
    OVERVIEW,
    OVERVIEW_COORDS,
    FOCUS_ALTITUDE,
    FLY_MS,
  };
}
