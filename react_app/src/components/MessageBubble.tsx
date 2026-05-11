import { useEffect, useRef, useState } from 'react';

import MessageContent, { getMessagePreviewText } from './MessageContent';
import type { MessageRecord } from '../types';

interface MessageBubbleProps {
  messageIndex: number;
  record: MessageRecord;
  deferContent?: boolean;
  onCopy: (text: string) => void;
  onDelete: (messageIndex: number) => void;
  onEdit: (messageIndex: number) => void;
  onRegenerate: (sourceText: string) => void;
}

export default function MessageBubble({
  messageIndex,
  record,
  deferContent = false,
  onCopy,
  onDelete,
  onEdit,
  onRegenerate,
}: MessageBubbleProps) {
  const messageRef = useRef<HTMLDivElement>(null);
  const [visibleRecord, setVisibleRecord] = useState<MessageRecord | null>(() => (
    deferContent ? null : record
  ));
  const canShowActions = !record.pending;
  const shouldDeferContent = deferContent && !record.pending;
  const shouldRenderContent = !shouldDeferContent || visibleRecord === record;

  useEffect(() => {
    if (!shouldDeferContent) {
      setVisibleRecord(record);
      return undefined;
    }

    const node = messageRef.current;
    if (!node || typeof IntersectionObserver === 'undefined') {
      setVisibleRecord(record);
      return undefined;
    }

    setVisibleRecord((previous) => (previous === record ? previous : null));

    const root = node.closest('.message-list') || null;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisibleRecord(record);
          observer.disconnect();
        }
      },
      {
        root,
        rootMargin: '900px 0px',
      },
    );

    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, [record, shouldDeferContent]);

  return (
    <div
      className={`message ${record.role}${record.pending ? ' pending' : ''}`}
      data-message-index={messageIndex}
      ref={messageRef}
    >
      <div className="message-body">
        <div className="content">
          {shouldRenderContent ? (
            <MessageContent record={record} />
          ) : (
            <div className="message-deferred-preview">{getMessagePreviewText(record)}</div>
          )}
        </div>
        {canShowActions ? (
          <div className="message-actions">
            <button className="action-btn" type="button" onClick={() => onCopy(record.text)}>
              <i className="ph-light ph-copy" />
            </button>
            {record.role === 'user' ? (
              <button className="action-btn" type="button" onClick={() => onEdit(messageIndex)}>
                <i className="ph-light ph-pencil-simple" />
              </button>
            ) : (
              <button className="action-btn" type="button" onClick={() => onRegenerate(record.sourceText)}>
                <i className="ph-light ph-arrows-clockwise" />
              </button>
            )}
            <button className="action-btn" type="button" onClick={() => onDelete(messageIndex)}>
              <i className="ph-light ph-trash" />
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
