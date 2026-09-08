import 'package:flutter/material.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  final List<Map<String, dynamic>> _mockAlerts = const [
    {
      "id": "IN-NER-SK-2026-0042",
      "severity": "EVACUATION",
      "title": "NH-10 Km 42-46 Immediate Evacuation",
      "desc": "Heavy antecedent rainfall (142mm/24h) triggered tension cracks. Factor of Safety is 0.84.",
      "instruction": "All traffic suspended. Move immediately to Govt Senior Secondary School.",
      "time": "15 mins ago",
      "color": Colors.redAccent
    },
    {
      "id": "IN-NER-ML-2026-0039",
      "severity": "WATCH",
      "title": "East Khasi Hills Road Slump Watch",
      "desc": "Soil saturation index reached 88%. Pore water pressures rising steadily.",
      "instruction": "Heavy transport vehicles restricted. Maintain safe braking distances.",
      "time": "2 hours ago",
      "color": Colors.orangeAccent
    },
    {
      "id": "IN-NER-AP-2026-0021",
      "severity": "ADVISORY",
      "title": "Papum Pare Hill Slope Weather Advisory",
      "desc": "Continuous moderate rain forecast for next 12 hours.",
      "instruction": "Stay alert during nighttime travel along valley cut roads.",
      "time": "5 hours ago",
      "color": Colors.amberAccent
    }
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF161B22),
        title: const Text("Emergency Early Warnings", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _mockAlerts.length,
        itemBuilder: (ctx, i) {
          final alert = _mockAlerts[i];
          final Color badgeColor = alert['color'] as Color;

          return Container(
            margin: const EdgeInsets.only(bottom: 16),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xFF161B22),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: badgeColor.withOpacity(0.4)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: badgeColor.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: badgeColor),
                      ),
                      child: Text(
                        alert['severity'] as String,
                        style: TextStyle(color: badgeColor, fontSize: 10, fontWeight: FontWeight.bold),
                      ),
                    ),
                    Text(alert['time'] as String, style: const TextStyle(color: Colors.white38, fontSize: 11)),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  alert['title'] as String,
                  style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 6),
                Text(alert['desc'] as String, style: const TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.black45,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.directions_run_rounded, color: Colors.amber, size: 18),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          alert['instruction'] as String,
                          style: const TextStyle(color: Colors.amberAccent, fontSize: 11, fontWeight: FontWeight.w500),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    TextButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.volume_up, size: 16),
                      label: const Text("Audio Alert", style: TextStyle(fontSize: 12)),
                      style: TextButton.styleFrom(foregroundColor: Colors.lightBlueAccent),
                    ),
                    const SizedBox(width: 8),
                    ElevatedButton.icon(
                      onPressed: () {},
                      icon: const Icon(Icons.map, size: 16),
                      label: const Text("View Map", style: TextStyle(fontSize: 12)),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: badgeColor,
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      ),
                    )
                  ],
                )
              ],
            ),
          );
        },
      ),
    );
  }
}
