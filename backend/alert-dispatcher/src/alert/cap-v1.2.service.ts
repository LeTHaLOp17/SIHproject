/**
 * OASIS Common Alerting Protocol (CAP-IN v1.2) Serialization Service
 * Specifically compliant with NDMA SACHET and C-DAC Pan-India Early Warning Platform.
 */

export interface CapAlertPayload {
  alertId: string;
  sender: string;
  sentAt: string;
  status: 'Actual' | 'Exercise' | 'System' | 'Test';
  msgType: 'Alert' | 'Update' | 'Cancel';
  scope: 'Public' | 'Restricted' | 'Private';
  category: 'Geo' | 'Met' | 'Safety' | 'Rescue';
  event: string;
  urgency: 'Immediate' | 'Expected' | 'Future';
  severity: 'Extreme' | 'Severe' | 'Moderate' | 'Minor';
  certainty: 'Observed' | 'Likely' | 'Possible';
  eventCode: string;
  expiresAt: string;
  headline: string;
  description: string;
  instruction: string;
  areaDesc: string;
  circleCoordinates: string; // "lat,lon,radius_km"
  translations?: {
    assamese?: { headline: string; instruction: string };
    bengali?: { headline: string; instruction: string };
    hindi?: { headline: string; instruction: string };
    bodo?: { headline: string; instruction: string };
    khasi?: { headline: string; instruction: string };
  };
}

export class CapXmlSerializer {
  /**
   * Generates standard OASIS CAP v1.2 XML string.
   */
  public static serializeToXml(payload: CapAlertPayload): string {
    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>${payload.alertId}</identifier>
  <sender>${payload.sender}</sender>
  <sent>${payload.sentAt}</sent>
  <status>${payload.status}</status>
  <msgType>${payload.msgType}</msgType>
  <scope>${payload.scope}</scope>
  <codeName>NDMA_SACHET_INDIA</codeName>
  <info>
    <language>en-IN</language>
    <category>${payload.category}</category>
    <event>${payload.event}</event>
    <urgency>${payload.urgency}</urgency>
    <severity>${payload.severity}</severity>
    <certainty>${payload.certainty}</certainty>
    <eventCode>
      <valueName>NDMA_DISASTER_CODE</valueName>
      <value>${payload.eventCode}</value>
    </eventCode>
    <expires>${payload.expiresAt}</expires>
    <senderName>State Disaster Management Authority (NER, MDoNER)</senderName>
    <headline><![CDATA[${payload.headline}]]></headline>
    <description><![CDATA[${payload.description}]]></description>
    <instruction><![CDATA[${payload.instruction}]]></instruction>
    <area>
      <areaDesc>${payload.areaDesc}</areaDesc>
      <circle>${payload.circleCoordinates}</circle>
    </area>
  </info>
  ${payload.translations?.assamese ? `
  <info>
    <language>as-IN</language>
    <category>${payload.category}</category>
    <event>ভূমিস্খলনৰ সতৰ্কবাণী</event>
    <urgency>${payload.urgency}</urgency>
    <severity>${payload.severity}</severity>
    <certainty>${payload.certainty}</certainty>
    <headline><![CDATA[${payload.translations.assamese.headline}]]></headline>
    <instruction><![CDATA[${payload.translations.assamese.instruction}]]></instruction>
    <area>
      <areaDesc>${payload.areaDesc}</areaDesc>
      <circle>${payload.circleCoordinates}</circle>
    </area>
  </info>` : ''}
  ${payload.translations?.hindi ? `
  <info>
    <language>hi-IN</language>
    <category>${payload.category}</category>
    <event>भूस्खलन चेतावनी</event>
    <urgency>${payload.urgency}</urgency>
    <severity>${payload.severity}</severity>
    <certainty>${payload.certainty}</certainty>
    <headline><![CDATA[${payload.translations.hindi.headline}]]></headline>
    <instruction><![CDATA[${payload.translations.hindi.instruction}]]></instruction>
    <area>
      <areaDesc>${payload.areaDesc}</areaDesc>
      <circle>${payload.circleCoordinates}</circle>
    </area>
  </info>` : ''}
</alert>`;
    return xml.trim();
  }
}
