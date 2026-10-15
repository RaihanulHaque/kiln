import NativeKiln from './NativeKiln';
import { KilnLibrary } from './kiln';

export * from './kiln';
export type { NativeDeviceInfo as DeviceInfo, NativeProjectorInfo as ProjectorInfo } from './NativeKiln';

/** The kiln library on this device. */
export const Kiln = new KilnLibrary(NativeKiln);
