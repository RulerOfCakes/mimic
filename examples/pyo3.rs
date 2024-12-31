use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let version: String = sys.getattr("version")?.extract()?;

        let locals = [("os", py.import("os")?)].into_py_dict(py)?;
        let code = c_str!("os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'");
        let user: String = py.eval(code, None, Some(&locals))?.extract()?;

        let pytorch = py.import("torch")?;
        let cuda_available = pytorch.getattr("cuda")?.getattr("is_available")?.call0()?;

        println!("Hello {}, I'm Python {}", user, version);
        println!("CUDA available: {}", cuda_available);
        Ok(())
    })
}