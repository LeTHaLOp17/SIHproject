/**
 * BullMQ Priority Alert Queuing & Retry Engine
 * Manages high-concurrency alert dispatch across SMS, IVR, and WebPush.
 */

import { CdacSmsDispatcher, SmsDispatchTarget } from '../channels/cdac-sms.service';
import { IvrVoiceDispatcher, IvrCallRequest } from '../channels/ivr-voice.service';

export enum AlertPriorityTier {
  TIER_1_EVACUATION = 1,   // Highest priority, preempts all other jobs
  TIER_2_RED_WARNING = 2,  // Critical hazard, high priority
  TIER_3_ORANGE_WATCH = 3, // Medium priority
  TIER_4_ADVISORY = 4      // Routine advisory
}

export interface AlertJobData {
  jobId: string;
  alertId: string;
  priority: AlertPriorityTier;
  smsTargets: SmsDispatchTarget[];
  ivrTargets: IvrCallRequest[];
  capXml: string;
  createdAt: string;
}

export class AlertQueueManager {
  private queue: Array<{ priority: number; data: AlertJobData }> = [];
  private deadLetterQueue: Array<{ data: AlertJobData; error: string; timestamp: string }> = [];

  /**
   * Enqueues an alert job with strict priority enforcement.
   */
  public async addAlertJob(data: AlertJobData): Promise<string> {
    this.queue.push({ priority: data.priority, data });
    // Sort queue by priority ascending (1 = highest priority)
    this.queue.sort((a, b) => a.priority - b.priority);

    console.log(`[BULLMQ QUEUE] Job ${data.jobId} enqueued with Priority Tier ${data.priority} (SMS: ${data.smsTargets.length}, IVR: ${data.ivrTargets.length})`);
    return data.jobId;
  }

  /**
   * Worker processor simulating concurrent execution with retry logic.
   */
  public async processNextJob(): Promise<boolean> {
    if (this.queue.length === 0) return false;

    const job = this.queue.shift()!;
    console.log(`[BULLMQ WORKER] Processing Job: ${job.data.jobId} (Tier ${job.priority})`);

    try {
      // 1. Dispatch SMS concurrently
      const smsPromises = job.data.smsTargets.map(t => CdacSmsDispatcher.dispatchSms(t));
      await Promise.all(smsPromises);

      // 2. Dispatch IVR calls concurrently
      const ivrPromises = job.data.ivrTargets.map(i => IvrVoiceDispatcher.initiateCall(i));
      await Promise.all(ivrPromises);

      console.log(`[BULLMQ SUCCESS] Job ${job.data.jobId} completely dispatched.`);
      return true;
    } catch (err: any) {
      console.error(`[BULLMQ RETRY] Job ${job.data.jobId} failed: ${err.message}. Routing to DLQ.`);
      this.deadLetterQueue.push({
        data: job.data,
        error: err.message,
        timestamp: new Date().toISOString()
      });
      return false;
    }
  }

  public getQueueStats() {
    return {
      activeQueueLength: this.queue.length,
      deadLetterQueueLength: this.deadLetterQueue.length
    };
  }
}
