// fourwing-viz — interactive volume viewer for fourwing7 escape-time data
//
// Loads .fw7vol binary volumes produced by: fourwing7 --dump-volume <path>
// Displays three orthogonal slices with interactive navigation.
//
// Controls:
//   Slider / scroll wheel  — move slice along its fixed axis
//   Click on a slice       — cross-navigate the other two axes
//   Colormap / Scale       — top bar selectors

use eframe::egui;
use std::path::{Path, PathBuf};

// ═══════════════════════════════════════════════════════════════════════════
// Data types
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum System {
    Fourwing,
    Chen,
}

impl std::fmt::Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fourwing => write!(f, "Fourwing"),
            Self::Chen => write!(f, "Chen"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Axis {
    A,
    B,
    C,
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "a"),
            Self::B => write!(f, "b"),
            Self::C => write!(f, "c"),
        }
    }
}

#[derive(Debug, Clone)]
struct AxisRange {
    min: f64,
    max: f64,
    count: u32,
}

impl AxisRange {
    fn value_at(&self, index: u32) -> f64 {
        if self.count <= 1 {
            return self.min;
        }
        let frac = index as f64 / (self.count - 1) as f64;
        self.min + (self.max - self.min) * frac
    }
}

#[derive(Debug, Clone)]
struct VolumeHeader {
    system: System,
    a: AxisRange,
    b: AxisRange,
    c: AxisRange,
    dt: f64,
    max_steps: u32,
    escape_radius: f64,
    seed_axis_count: u32,
}

struct Volume {
    header: VolumeHeader,
    data: Vec<u32>,
}

impl Volume {
    fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if bytes.len() < 132 {
            return Err("file too small for FW7VOL1 header".into());
        }
        if &bytes[0..7] != b"FW7VOL1" {
            return Err("bad magic: expected FW7VOL1".into());
        }

        let mut o = 8usize;
        let _version = read_u32(&bytes, &mut o);
        let sys_id = read_u32(&bytes, &mut o);
        let ac = read_u32(&bytes, &mut o);
        let bc = read_u32(&bytes, &mut o);
        let cc = read_u32(&bytes, &mut o);
        let a_min = read_f64(&bytes, &mut o);
        let a_max = read_f64(&bytes, &mut o);
        let b_min = read_f64(&bytes, &mut o);
        let b_max = read_f64(&bytes, &mut o);
        let c_min = read_f64(&bytes, &mut o);
        let c_max = read_f64(&bytes, &mut o);
        let dt = read_f64(&bytes, &mut o);
        let max_steps = read_u32(&bytes, &mut o);
        let escape_radius = read_f64(&bytes, &mut o);
        let seed_axis_count = read_u32(&bytes, &mut o);
        o += 32; // layout[16] + reserved[16]

        let system = match sys_id {
            1 => System::Fourwing,
            2 => System::Chen,
            _ => return Err(format!("unknown system id: {sys_id}")),
        };

        let total = ac as usize * bc as usize * cc as usize;
        if bytes.len() < o + total * 4 {
            return Err("file truncated: not enough voxel data".into());
        }

        let mut data = Vec::with_capacity(total);
        for i in 0..total {
            let b = o + i * 4;
            data.push(u32::from_le_bytes([
                bytes[b],
                bytes[b + 1],
                bytes[b + 2],
                bytes[b + 3],
            ]));
        }

        Ok(Self {
            header: VolumeHeader {
                system,
                a: AxisRange { min: a_min, max: a_max, count: ac },
                b: AxisRange { min: b_min, max: b_max, count: bc },
                c: AxisRange { min: c_min, max: c_max, count: cc },
                dt,
                max_steps,
                escape_radius,
                seed_axis_count,
            },
            data,
        })
    }

    #[inline]
    fn get(&self, ia: u32, ib: u32, ic: u32) -> u32 {
        let idx = ((ia as usize) * self.header.b.count as usize + ib as usize)
            * self.header.c.count as usize
            + ic as usize;
        self.data[idx]
    }

    /// Sample a 2D slice with one axis held fixed.
    /// Rows and columns follow the same convention as the C++ PGM writer.
    fn slice_value(&self, axis: Axis, fixed: u32, row: u32, col: u32) -> u32 {
        match axis {
            Axis::A => self.get(fixed, row, col), // rows=b, cols=c
            Axis::B => self.get(row, fixed, col),  // rows=a, cols=c
            Axis::C => self.get(row, col, fixed),  // rows=a, cols=b
        }
    }

    /// (width, height) of a 2D slice orthogonal to `axis`.
    fn slice_dims(&self, axis: Axis) -> (u32, u32) {
        match axis {
            Axis::A => (self.header.c.count, self.header.b.count),
            Axis::B => (self.header.c.count, self.header.a.count),
            Axis::C => (self.header.b.count, self.header.a.count),
        }
    }

    fn axis_range(&self, axis: Axis) -> &AxisRange {
        match axis {
            Axis::A => &self.header.a,
            Axis::B => &self.header.b,
            Axis::C => &self.header.c,
        }
    }
}

fn read_u32(d: &[u8], o: &mut usize) -> u32 {
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    v
}

fn read_f64(d: &[u8], o: &mut usize) -> f64 {
    let v = f64::from_le_bytes(d[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

// ═══════════════════════════════════════════════════════════════════════════
// Colormaps & scaling
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Colormap {
    Inferno,
    Viridis,
    Magma,
    Plasma,
    Turbo,
    Grayscale,
}

const ALL_COLORMAPS: &[Colormap] = &[
    Colormap::Inferno,
    Colormap::Viridis,
    Colormap::Magma,
    Colormap::Plasma,
    Colormap::Turbo,
    Colormap::Grayscale,
];

impl Colormap {
    fn name(self) -> &'static str {
        match self {
            Self::Inferno => "Inferno",
            Self::Viridis => "Viridis",
            Self::Magma => "Magma",
            Self::Plasma => "Plasma",
            Self::Turbo => "Turbo",
            Self::Grayscale => "Grayscale",
        }
    }

    fn eval(self, t: f64) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        if self == Self::Grayscale {
            let v = (t * 255.0) as u8;
            return [v, v, v];
        }
        let gradient = match self {
            Self::Inferno => colorous::INFERNO,
            Self::Viridis => colorous::VIRIDIS,
            Self::Magma => colorous::MAGMA,
            Self::Plasma => colorous::PLASMA,
            Self::Turbo => colorous::TURBO,
            Self::Grayscale => unreachable!(),
        };
        let c = gradient.eval_continuous(t);
        [c.r, c.g, c.b]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scale {
    Linear,
    Log,
}

fn normalize(val: u32, max: u32, scale: Scale) -> f64 {
    if max == 0 {
        return 0.0;
    }
    match scale {
        Scale::Linear => val as f64 / max as f64,
        Scale::Log => {
            if val == 0 {
                0.0
            } else {
                (1.0 + val as f64).ln() / (1.0 + max as f64).ln()
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Image generation
// ═══════════════════════════════════════════════════════════════════════════

fn make_slice_image(
    vol: &Volume,
    axis: Axis,
    fixed: u32,
    cm: Colormap,
    sc: Scale,
) -> egui::ColorImage {
    let (w, h) = vol.slice_dims(axis);
    let mut pixels = Vec::with_capacity((w * h) as usize);
    for row in 0..h {
        for col in 0..w {
            let v = vol.slice_value(axis, fixed, row, col);
            let t = normalize(v, vol.header.max_steps, sc);
            let [r, g, b] = cm.eval(t);
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }
    egui::ColorImage {
        size: [w as usize, h as usize],
        pixels,
    }
}

fn make_colorbar(cm: Colormap, width: usize) -> egui::ColorImage {
    let mut pixels = Vec::with_capacity(width);
    for i in 0..width {
        let t = i as f64 / (width - 1).max(1) as f64;
        let [r, g, b] = cm.eval(t);
        pixels.push(egui::Color32::from_rgb(r, g, b));
    }
    egui::ColorImage {
        size: [width, 1],
        pixels,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Application
// ═══════════════════════════════════════════════════════════════════════════

/// Cached state for one slice view.
struct SliceView {
    tex: Option<egui::TextureHandle>,
    idx: u32,
    prev_idx: u32,
    prev_cm: Colormap,
    prev_sc: Scale,
}

impl SliceView {
    fn new(idx: u32) -> Self {
        Self {
            tex: None,
            idx,
            prev_idx: u32::MAX,
            prev_cm: Colormap::Grayscale,
            prev_sc: Scale::Linear,
        }
    }

    fn dirty(&self, cm: Colormap, sc: Scale) -> bool {
        self.tex.is_none()
            || self.idx != self.prev_idx
            || cm != self.prev_cm
            || sc != self.prev_sc
    }
}

struct Viewer {
    vol: Option<Volume>,
    err: Option<String>,
    views: [SliceView; 3], // indices: 0=A, 1=B, 2=C
    cm: Colormap,
    sc: Scale,
    cbar_tex: Option<egui::TextureHandle>,
    cbar_cm: Colormap,
    hover_text: String,
}

impl Default for Viewer {
    fn default() -> Self {
        Self {
            vol: None,
            err: None,
            views: [SliceView::new(64), SliceView::new(64), SliceView::new(64)],
            cm: Colormap::Inferno,
            sc: Scale::Log,
            cbar_tex: None,
            cbar_cm: Colormap::Grayscale,
            hover_text: String::new(),
        }
    }
}

impl Viewer {
    fn load(&mut self, path: &Path) {
        match Volume::load(path) {
            Ok(v) => {
                self.views[0].idx = v.header.a.count / 2;
                self.views[1].idx = v.header.b.count / 2;
                self.views[2].idx = v.header.c.count / 2;
                self.vol = Some(v);
                self.err = None;
            }
            Err(e) => {
                self.err = Some(e);
                self.vol = None;
            }
        }
    }

    fn refresh_textures(&mut self, ctx: &egui::Context) {
        const AXES: [Axis; 3] = [Axis::A, Axis::B, Axis::C];
        for (i, &ax) in AXES.iter().enumerate() {
            if !self.views[i].dirty(self.cm, self.sc) {
                continue;
            }
            if let Some(vol) = &self.vol {
                let img = make_slice_image(vol, ax, self.views[i].idx, self.cm, self.sc);
                match &mut self.views[i].tex {
                    Some(t) => t.set(img, egui::TextureOptions::NEAREST),
                    None => {
                        self.views[i].tex = Some(ctx.load_texture(
                            format!("slice_{ax}"),
                            img,
                            egui::TextureOptions::NEAREST,
                        ));
                    }
                }
                self.views[i].prev_idx = self.views[i].idx;
                self.views[i].prev_cm = self.cm;
                self.views[i].prev_sc = self.sc;
            }
        }

        // Colorbar
        if self.cbar_tex.is_none() || self.cbar_cm != self.cm {
            let img = make_colorbar(self.cm, 256);
            match &mut self.cbar_tex {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => {
                    self.cbar_tex =
                        Some(ctx.load_texture("cbar", img, egui::TextureOptions::LINEAR));
                }
            }
            self.cbar_cm = self.cm;
        }
    }
}

/// Labels for each slice panel: (fixed axis name, row axis name, col axis name)
const SLICE_LABELS: [(&str, &str, &str); 3] = [
    ("a", "b", "c"),
    ("b", "a", "c"),
    ("c", "a", "b"),
];

const AXES: [Axis; 3] = [Axis::A, Axis::B, Axis::C];

impl eframe::App for Viewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_textures(ctx);

        // Accept dropped files
        ctx.input(|inp| {
            if let Some(f) = inp.raw.dropped_files.first() {
                if let Some(p) = &f.path {
                    self.load(p);
                }
            }
        });

        // ── Top bar ──────────────────────────────────────────────────────
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("fourwing7").strong().size(16.0));
                ui.separator();

                ui.label("Colormap");
                egui::ComboBox::from_id_salt("cm_sel")
                    .selected_text(self.cm.name())
                    .show_ui(ui, |ui| {
                        for &c in ALL_COLORMAPS {
                            ui.selectable_value(&mut self.cm, c, c.name());
                        }
                    });

                ui.separator();
                ui.label("Scale");
                ui.selectable_value(&mut self.sc, Scale::Linear, "Linear");
                ui.selectable_value(&mut self.sc, Scale::Log, "Log");
            });
        });

        // ── Bottom bar ───────────────────────────────────────────────────
        egui::TopBottomPanel::bottom("info")
            .min_height(44.0)
            .show(ctx, |ui| {
                // Colorbar strip
                if let Some(tex) = &self.cbar_tex {
                    let w = (ui.available_width() - 100.0).max(60.0);
                    ui.horizontal(|ui| {
                        ui.label("0");
                        let sized = egui::load::SizedTexture::new(tex.id(), [w, 10.0]);
                        ui.image(sized);
                        if let Some(v) = &self.vol {
                            ui.label(format!("{}", v.header.max_steps));
                        }
                    });
                }

                // Metadata line
                ui.horizontal(|ui| {
                    if let Some(v) = &self.vol {
                        let h = &v.header;
                        ui.label(format!(
                            "{} | {}x{}x{} | dt={:.4} | esc_r={:.0} | seeds={}^3",
                            h.system,
                            h.a.count,
                            h.b.count,
                            h.c.count,
                            h.dt,
                            h.escape_radius,
                            h.seed_axis_count,
                        ));
                    }
                    if !self.hover_text.is_empty() {
                        ui.separator();
                        ui.label(&self.hover_text);
                    }
                });
            });

        // ── Central panel: three slice views ─────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.vol.is_none() {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    if let Some(e) = &self.err {
                        ui.colored_label(egui::Color32::from_rgb(255, 80, 80), e);
                        ui.add_space(20.0);
                    }
                    ui.heading("fourwing7 volume viewer");
                    ui.add_space(12.0);
                    ui.label("Drop a .fw7vol file here, or pass it as a CLI argument.");
                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("Generate a volume with:").weak());
                    ui.monospace("  ./fourwing7 --system chen --dump-volume chen.fw7vol");
                    ui.monospace("  ./fourwing7 --system fourwing --dump-volume fourwing.fw7vol");
                });
                return;
            }

            self.hover_text.clear();

            let avail = ui.available_size();
            let img_sz = ((avail.x - 48.0) / 3.0).min(avail.y - 64.0).max(80.0);

            ui.horizontal(|ui| {
                for i in 0..3 {
                    self.draw_slice_panel(ui, ctx, i, img_sz);
                    if i < 2 {
                        ui.separator();
                    }
                }
            });
        });
    }
}

impl Viewer {
    fn draw_slice_panel(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        i: usize,
        img_sz: f32,
    ) {
        let ax = AXES[i];
        let (fix_label, row_label, col_label) = SLICE_LABELS[i];

        let vol = self.vol.as_ref().unwrap();
        let range = vol.axis_range(ax);
        let max_idx = range.count.saturating_sub(1);
        let val = range.value_at(self.views[i].idx);

        ui.vertical(|ui| {
            // Header
            ui.label(
                egui::RichText::new(format!("{fix_label} = {val:.3}  [{row_label} vs {col_label}]"))
                    .strong(),
            );

            // Slice image
            if let Some(tex) = &self.views[i].tex {
                let desired = egui::vec2(img_sz, img_sz);
                let (rect, resp) = ui.allocate_exact_size(desired, egui::Sense::click());

                if ui.is_rect_visible(rect) {
                    // Draw the image
                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    ui.painter()
                        .image(tex.id(), rect, uv, egui::Color32::WHITE);

                    // Draw crosshairs showing where the other two slices cut through
                    let vol = self.vol.as_ref().unwrap();
                    let (w, h) = vol.slice_dims(ax);
                    let (cross_row, cross_col) = match ax {
                        Axis::A => (self.views[1].idx, self.views[2].idx),
                        Axis::B => (self.views[0].idx, self.views[2].idx),
                        Axis::C => (self.views[0].idx, self.views[1].idx),
                    };
                    let cx = rect.min.x + (cross_col as f32 + 0.5) / w as f32 * rect.width();
                    let cy = rect.min.y + (cross_row as f32 + 0.5) / h as f32 * rect.height();
                    let stroke = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(90));
                    ui.painter().line_segment(
                        [egui::pos2(cx, rect.min.y), egui::pos2(cx, rect.max.y)],
                        stroke,
                    );
                    ui.painter().line_segment(
                        [egui::pos2(rect.min.x, cy), egui::pos2(rect.max.x, cy)],
                        stroke,
                    );
                }

                // Scroll wheel: move slice index
                if resp.hovered() {
                    let delta = ctx.input(|inp| inp.smooth_scroll_delta.y);
                    if delta.abs() > 1.0 {
                        let d: i32 = if delta > 0.0 { -1 } else { 1 };
                        self.views[i].idx =
                            (self.views[i].idx as i32 + d).clamp(0, max_idx as i32) as u32;
                    }
                }

                // Hover info and click-to-navigate
                if resp.hovered() || resp.clicked() {
                    if let Some(pos) = ctx.input(|inp| inp.pointer.hover_pos()) {
                        if rect.contains(pos) {
                            let vol = self.vol.as_ref().unwrap();
                            let (w, h) = vol.slice_dims(ax);
                            let lx = pos.x - rect.min.x;
                            let ly = pos.y - rect.min.y;
                            let col = ((lx / rect.width()) * w as f32)
                                .clamp(0.0, (w - 1) as f32)
                                as u32;
                            let row = ((ly / rect.height()) * h as f32)
                                .clamp(0.0, (h - 1) as f32)
                                as u32;
                            let steps = vol.slice_value(ax, self.views[i].idx, row, col);

                            let (row_range, col_range) = match ax {
                                Axis::A => (&vol.header.b, &vol.header.c),
                                Axis::B => (&vol.header.a, &vol.header.c),
                                Axis::C => (&vol.header.a, &vol.header.b),
                            };
                            self.hover_text = format!(
                                "{row_label}={:.3}  {col_label}={:.3}  |  steps={steps}",
                                row_range.value_at(row),
                                col_range.value_at(col),
                            );

                            // Click: set the other two slice indices to the clicked position
                            if resp.clicked() {
                                match ax {
                                    Axis::A => {
                                        self.views[1].idx = row; // b
                                        self.views[2].idx = col; // c
                                    }
                                    Axis::B => {
                                        self.views[0].idx = row; // a
                                        self.views[2].idx = col; // c
                                    }
                                    Axis::C => {
                                        self.views[0].idx = row; // a
                                        self.views[1].idx = col; // b
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Slider
            let mut f = self.views[i].idx as f32;
            if ui
                .add(
                    egui::Slider::new(&mut f, 0.0..=max_idx as f32)
                        .step_by(1.0)
                        .show_value(false)
                        .text(format!("[{}/{}]", self.views[i].idx, max_idx)),
                )
                .changed()
            {
                self.views[i].idx = f as u32;
            }
        });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("fourwing-viz — interactive volume viewer for fourwing7 escape-time data");
        eprintln!();
        eprintln!("Usage: fourwing-viz [volume.fw7vol]");
        eprintln!();
        eprintln!("  Supports drag-and-drop. Scroll over slices to navigate.");
        eprintln!("  Click a slice to cross-navigate the other two axes.");
        eprintln!();
        eprintln!("Generate a volume:");
        eprintln!("  ./fourwing7 --system chen --dump-volume chen.fw7vol");
        eprintln!("  ./fourwing7 --system fourwing --dump-volume fourwing.fw7vol");
        std::process::exit(0);
    }

    let path = args.get(1).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("fourwing7 viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "fourwing7",
        options,
        Box::new(move |_cc| {
            let mut app = Viewer::default();
            if let Some(p) = &path {
                app.load(p);
            }
            Ok(Box::new(app))
        }),
    )
}
