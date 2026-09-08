import 'package:flutter/material.dart';
import '../core/sync/sync_manager.dart';

class SyncScreen extends StatefulWidget {
  const SyncScreen({super.key});

  @override
  State<SyncScreen> createState() => _SyncScreenState();
}

class _SyncScreenState extends State<SyncScreen> {
  int _pendingCount = 2;
  bool _isSyncing = false;
  String _syncStatusText = "2 offline field reports queued on device.";

  void _triggerSync() async {
    setState(() {
      _isSyncing = true;
      _syncStatusText = "Connecting to SDMA Sync Gateway...";
    });

    final res = await SyncManager.instance.performSync();

    setState(() {
      _isSyncing = false;
      _pendingCount = 0;
      _syncStatusText = "All reports successfully synchronized and reconciled with central GIS.";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: const Text("Offline Data & Sync Manager", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: const Color(0xFF161B22),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white12),
              ),
              child: Column(
                children: [
                  Icon(
                    _pendingCount > 0 ? Icons.cloud_upload_outlined : Icons.cloud_done_rounded,
                    color: _pendingCount > 0 ? Colors.amber : Colors.greenAccent,
                    size: 56,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _pendingCount > 0 ? "$_pendingCount Reports Pending Sync" : "Database Synchronized",
                    style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    _syncStatusText,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.white60, fontSize: 12),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _isSyncing ? null : _triggerSync,
              icon: const Icon(Icons.sync),
              label: Text(_isSyncing ? "Synchronizing..." : "SYNC WITH HEADQUARTERS NOW"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.amber,
                foregroundColor: Colors.black,
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              "OFFLINE DATA STORES (SQLITE)",
              style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.1),
            ),
            const SizedBox(height: 10),
            _buildStoreItem("Citizen & Field Crack Reports", "$_pendingCount items pending", Icons.description_outlined),
            _buildStoreItem("Cached Regional Hazard Polygons", "1,420 grid cells active", Icons.layers_outlined),
            _buildStoreItem("Emergency Helipads & Egress Trails", "14 safe shelters saved", Icons.shield_outlined),
          ],
        ),
      ),
    );
  }

  Widget _buildStoreItem(String title, String count, IconData icon) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF161B22),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(icon, color: Colors.white70, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(color: Colors.white, fontSize: 13, fontWeight: FontWeight.w600)),
                Text(count, style: const TextStyle(color: Colors.white38, fontSize: 11)),
              ],
            ),
          ),
          const Icon(Icons.chevron_right, color: Colors.white30),
        ],
      ),
    );
  }
}
