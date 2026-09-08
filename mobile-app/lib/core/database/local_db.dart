/**
 * Offline-First SQLite Local Storage Database
 * Stores local reports, offline hazard polygons, and active alerts on the phone.
 */

import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class LocalDatabaseHelper {
  static final LocalDatabaseHelper instance = LocalDatabaseHelper._init();
  static Database? _database;

  LocalDatabaseHelper._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('ner_landslide_local.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    return await openDatabase(
      path,
      version: 1,
      onCreate: _createDB,
    );
  }

  Future _createDB(Database db, int version) async {
    // 1. Offline Citizen & Field Reports
    await db.execute('''
      CREATE TABLE offline_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        local_uuid TEXT NOT NULL,
        observation_type TEXT NOT NULL,
        apparent_crack_width_cm REAL NOT NULL,
        description TEXT,
        image_path TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        local_tinyml_risk REAL NOT NULL,
        created_at TEXT NOT NULL,
        sync_status TEXT NOT NULL DEFAULT 'PENDING'
      )
    ''');

    // 2. Offline Cached Emergency Alerts
    await db.execute('''
      CREATE TABLE cached_alerts (
        alert_id TEXT PRIMARY KEY,
        severity TEXT NOT NULL,
        headline TEXT NOT NULL,
        instruction TEXT NOT NULL,
        affected_area TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        received_at TEXT NOT NULL
      )
    ''');

    // 3. Pre-downloaded Village Evacuation Routes
    await db.execute('''
      CREATE TABLE village_routes (
        village_id TEXT PRIMARY KEY,
        village_name TEXT NOT NULL,
        shelter_name TEXT NOT NULL,
        shelter_lat REAL NOT NULL,
        shelter_lon REAL NOT NULL,
        route_geojson TEXT NOT NULL
      )
    ''');
  }

  Future<int> insertReport(Map<String, dynamic> row) async {
    final db = await instance.database;
    return await db.insert('offline_reports', row);
  }

  Future<List<Map<String, dynamic>>> getPendingReports() async {
    final db = await instance.database;
    return await db.query(
      'offline_reports',
      where: 'sync_status = ?',
      whereArgs: ['PENDING'],
      orderBy: 'created_at ASC',
    );
  }

  Future<int> markReportSynced(String localUuid) async {
    final db = await instance.database;
    return await db.update(
      'offline_reports',
      {'sync_status': 'SYNCED'},
      where: 'local_uuid = ?',
      whereArgs: [localUuid],
    );
  }

  Future<int> cacheAlert(Map<String, dynamic> alert) async {
    final db = await instance.database;
    return await db.insert('cached_alerts', alert, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  Future<List<Map<String, dynamic>>> getActiveAlerts() async {
    final db = await instance.database;
    return await db.query('cached_alerts', orderBy: 'received_at DESC');
  }
}
