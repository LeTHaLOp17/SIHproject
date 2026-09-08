/**
 * NestJS Alert Dispatcher Entry Point
 * Demonstrates CAP-IN v1.2 generation, BullMQ priority enqueueing, and multi-channel delivery.
 */

import { CapXmlSerializer, CapAlertPayload } from './alert/cap-v1.2.service';
import { AlertQueueManager, AlertPriorityTier, AlertJobData } from './queue/alert.queue';

async function bootstrap() {
  console.log("================================================================");
  console.log("🚀 STARTING MDoNER LANDSLIDE ALERT DISPATCHER (NESTJS MICROSERVICE)");
  console.log("================================================================");

  const queueManager = new AlertQueueManager();

  // 1. Prepare sample CAP-IN v1.2 Alert Payload for NH-10 Corridor
  const sampleCapPayload: CapAlertPayload = {
    alertId: `IN-NER-SK-DDMA-${Date.now()}`,
    sender: "ddma.gangtok@sikkim.gov.in",
    sentAt: new Date().toISOString(),
    status: "Actual",
    msgType: "Alert",
    scope: "Public",
    category: "Geo",
    event: "Immediate Landslide Warning",
    urgency: "Immediate",
    severity: "Extreme",
    certainty: "Observed",
    eventCode: "LS-WARN-RED",
    expiresAt: new Date(Date.now() + 12 * 3600 * 1000).toISOString(),
    headline: "IMMEDIATE EVACUATION: Critical Landslide Threat along NH-10 Corridor",
    description: "Heavy rainfall (142mm/24h) and pore-pressure sensors indicate immediate hillslope failure between KM 42 and KM 46. Factor of Safety is 0.84.",
    instruction: "All vehicular traffic on NH-10 is suspended. Residents of Rongli Upper Basti are instructed to move immediately to designated higher ground.",
    areaDesc: "NH-10 Km 42-46, East Sikkim",
    circleCoordinates: "27.3389,88.6065,2.5",
    translations: {
      assamese: {
        headline: "জৰুৰী সতৰ্কবাণী: এনএইচ-১০ কৰিডৰত ভূমিস্খলনৰ প্ৰচণ্ড বিপদ",
        instruction: "ৰংলি উচ্চ বস্তিৰ বাসিন্দাসকলক অনতিপলমে সুৰক্ষিত আশ্ৰয়স্থললৈ স্থানান্তৰিত হ'বলৈ নিৰ্দেশ দিয়া হৈছে।"
      },
      hindi: {
        headline: "तत्काल निकासी: NH-10 कॉरिडोर पर भीषण भूस्खलन की चेतावनी",
        instruction: "रोंगली अपर बस्ती के निवासियों को तुरंत सुरक्षित उच्च स्थान पर जाने का निर्देश दिया जाता है।"
      }
    }
  };

  // 2. Generate CAP XML
  const capXml = CapXmlSerializer.serializeToXml(sampleCapPayload);
  console.log("\n[CAP v1.2 XML GENERATED] Snippet:\n" + capXml.slice(0, 380) + "...\n");

  // 3. Enqueue High-Priority Job into BullMQ
  const job: AlertJobData = {
    jobId: `JOB-${Date.now()}`,
    alertId: sampleCapPayload.alertId,
    priority: AlertPriorityTier.TIER_1_EVACUATION,
    smsTargets: [
      {
        phoneNumber: "+919876543210",
        language: "as",
        villageName: "Rongli Upper Basti",
        hazardSeverity: "EVACUATION",
        safeShelterName: "Govt Senior Secondary School"
      },
      {
        phoneNumber: "+919876543211",
        language: "hi",
        villageName: "Rongli Upper Basti",
        hazardSeverity: "EVACUATION",
        safeShelterName: "Govt Senior Secondary School"
      }
    ],
    ivrTargets: [
      {
        callId: `CALL-01`,
        phoneNumber: "+919876543210",
        language: "as",
        villageName: "Rongli Upper Basti",
        hazardLevel: "RED_EVACUATION",
        maxRetryAttempts: 3
      }
    ],
    capXml,
    createdAt: new Date().toISOString()
  };

  await queueManager.addAlertJob(job);

  // 4. Process the Job
  console.log("\n[WORKER DISPATCH TEST]");
  await queueManager.processNextJob();

  console.log("\n✅ ALERT DISPATCHER MICROSERVICE TEST COMPLETED SUCCESSFULLY.");
}

if (require.main === module) {
  bootstrap();
}
