import 'package:flutter/material.dart';
import '../core/database/local_db.dart';
import '../ml/tinyml_inference.dart';

class ReportScreen extends StatefulWidget {
  const ReportScreen({super.key});

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  String _selectedObservation = 'TENSION_CRACK_ON_ROAD';
  double _crackWidthCm = 12.0;
  bool _muddySpringWater = true;
  bool _leaningTrees = false;
  final TextEditingController _descController = TextEditingController();
  bool _isSaving = false;

  void _saveOfflineReport() async {
    setState(() => _isSaving = true);

    // 1. Run local edge AI inference on the observation
    final eval = TinyMLInferenceEngine.instance.runLocalInference(
      slopeDegrees: 38.0,
      estimatedSoilDepthM: 2.5,
      rainfallHoursIntensityMm: 25.0,
      apparentCrackWidthCm: _crackWidthCm,
      isSpringDischargeMuddy: _muddySpringWater,
    );

    // 2. Persist to local SQLite offline table
    final row = {
      'local_uuid': 'REP-${DateTime.now().millisecondsSinceEpoch}',
      'observation_type': _selectedObservation,
      'apparent_crack_width_cm': _crackWidthCm,
      'description': _descController.text,
      'image_path': '/storage/emulated/0/DCIM/CRACK_001.jpg',
      'latitude': 27.3389,
      'longitude': 88.6065,
      'local_tinyml_risk': eval.failureProbability,
      'created_at': DateTime.now().toIso8601String(),
      'sync_status': 'PENDING'
    };

    await LocalDatabaseHelper.instance.insertReport(row);

    setState(() => _isSaving = false);

    if (!mounted) return;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF161B22),
        title: const Row(
          children: [
            Icon(Icons.check_circle, color: Colors.greenAccent),
            SizedBox(width: 8),
            Text("Report Saved Offline", style: TextStyle(color: Colors.white, fontSize: 16)),
          ],
        ),
        content: Text(
          "Your observation was stored in local SQLite database.\n\nLocal TinyML Risk Score: ${(eval.failureProbability * 100).toStringAsFixed(0)}%\n\nThis report will automatically sync with SDMA headquarters as soon as mobile or satellite connectivity is detected.",
          style: const TextStyle(color: Colors.white70, fontSize: 13),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              Navigator.pop(context);
            },
            child: const Text("OK", style: TextStyle(color: Colors.amber)),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: const Text("Report Ground Crack / Hazard", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Simulated Camera Viewfinder
            Container(
              height: 180,
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.amber.withOpacity(0.5)),
              ),
              child: Stack(
                alignment: Alignment.center,
                children: [
                  const Icon(Icons.camera_alt_outlined, color: Colors.white30, size: 54),
                  Positioned(
                    bottom: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(color: Colors.black87, borderRadius: BorderRadius.circular(6)),
                      child: const Text("On-Screen Metric Calibration: 1px = 0.28mm", style: TextStyle(color: Colors.amberAccent, fontSize: 11)),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            const Text("OBSERVATION TYPE", style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.1)),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(color: const Color(0xFF161B22), borderRadius: BorderRadius.circular(8)),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: _selectedObservation,
                  dropdownColor: const Color(0xFF161B22),
                  style: const TextStyle(color: Colors.white),
                  items: const [
                    DropdownMenuItem(value: 'TENSION_CRACK_ON_ROAD', child: Text("Tension Crack on Highway / Pavement")),
                    DropdownMenuItem(value: 'MUDDY_SPRING_WATER', child: Text("Fresh Muddy Water Springing from Slope")),
                    DropdownMenuItem(value: 'LEANING_TREES_OR_POLES', child: Text("Tilted Trees or Power Transmission Poles")),
                    DropdownMenuItem(value: 'ROAD_SURFACE_SUBSIDENCE', child: Text("Road Surface Sinking / Slump")),
                  ],
                  onChanged: (val) {
                    if (val != null) setState(() => _selectedObservation = val);
                  },
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Crack Width Slider
            Text("Apparent Crack Width: ${_crackWidthCm.toStringAsFixed(1)} cm", style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13)),
            Slider(
              value: _crackWidthCm,
              min: 0.5,
              max: 50.0,
              divisions: 99,
              activeColor: Colors.amber,
              onChanged: (val) => setState(() => _crackWidthCm = val),
            ),
            const SizedBox(height: 12),

            // Indicator Switches
            SwitchListTile(
              title: const Text("Muddy spring water emerging from slope toe", style: TextStyle(color: Colors.white, fontSize: 13)),
              subtitle: const Text("Strong signal of active subsurface pore pressure accumulation", style: TextStyle(color: Colors.white54, fontSize: 10)),
              value: _muddySpringWater,
              activeColor: Colors.amber,
              onChanged: (val) => setState(() => _muddySpringWater = val),
            ),
            SwitchListTile(
              title: const Text("Power lines or hillside trees visibly tilted", style: TextStyle(color: Colors.white, fontSize: 13)),
              value: _leaningTrees,
              activeColor: Colors.amber,
              onChanged: (val) => setState(() => _leaningTrees = val),
            ),
            const SizedBox(height: 20),

            // Description field
            TextField(
              controller: _descController,
              maxLines: 2,
              style: const TextStyle(color: Colors.white, fontSize: 13),
              decoration: InputDecoration(
                hintText: "Optional landmark (e.g. Near KM 42 culvert)",
                hintStyle: const TextStyle(color: Colors.white30),
                filled: true,
                fillColor: const Color(0xFF161B22),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
              ),
            ),
            const SizedBox(height: 24),

            ElevatedButton.icon(
              onPressed: _isSaving ? null : _saveOfflineReport,
              icon: const Icon(Icons.save_rounded),
              label: Text(_isSaving ? "Saving Locally..." : "SAVE & QUEUE OFFLINE REPORT"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.amber,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
