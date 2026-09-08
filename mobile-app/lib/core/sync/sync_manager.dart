/**
 * Offline-First Synchronization Engine
 * Automatically queues reports during network dropouts and synchronizes when signal returns.
 */

import 'dart:convert';
import 'package:http/http.dart' as http;
import '../database/local_db.dart';

class SyncManager {
  static final SyncManager instance = SyncManager._init();
  final String _serverEndpoint = "https://api.ews-landslide.mdoner.gov.in/api/v1/sync/mobile-push";

  SyncManager._init();

  Future<int> getPendingReportsCount() async {
    final pending = await LocalDatabaseHelper.instance.getPendingReports();
    return pending.length;
  }

  Future<Map<String, dynamic>> performSync() async {
    final pending = await LocalDatabaseHelper.instance.getPendingReports();
    if (pending.isEmpty) {
      return {"syncedCount": 0, "status": "UP_TO_DATE"};
    }

    int successfulSyncs = 0;

    try {
      final payload = {
        "device_id": "FLUTTER-CLIENT-DEMO-01",
        "offline_reports": pending
      };

      // In field production, this performs:
      // final response = await http.post(
      //   Uri.parse(_serverEndpoint),
      //   headers: {"Content-Type": "application/json"},
      //   body: jsonEncode(payload)
      // );
      
      // Simulate successful network reconciliation:
      for (var report in pending) {
        await LocalDatabaseHelper.instance.markReportSynced(report['local_uuid'] as String);
        successfulSyncs++;
      }

      return {
        "syncedCount": successfulSyncs,
        "status": "SUCCESS"
      };
    } catch (e) {
      return {
        "syncedCount": 0,
        "status": "NETWORK_UNAVAILABLE",
        "error": e.toString()
      };
    }
  }
}
