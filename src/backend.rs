//! Device selection is a build-time choice; CUDA remains the default.
use anyhow::Result;
use candle_core::Device;

pub fn new_device(index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Ok(Device::new_cuda(index)?)
    }
    #[cfg(feature = "mlx")]
    {
        anyhow::ensure!(
            index == 0,
            "Apple Silicon has one unified-memory GPU; use --device 0"
        );
        crate::apple::init()?;
        Ok(Device::new_metal(index)?)
    }
    #[cfg(not(any(feature = "cuda", feature = "mlx")))]
    {
        let _ = index;
        anyhow::bail!("build with the cuda or mlx feature")
    }
}
