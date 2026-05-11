import type { RefObject } from 'react';

import type { MessageBlock, MessageRecord } from '../types';
import MessageBubble from './MessageBubble';

interface MessageListProps {
  listRef: RefObject<HTMLDivElement | null>;
  messages: MessageRecord[];
  onCopy: (text: string) => void;
  onDeleteMessage: (messageIndex: number) => void;
  onEdit: (messageIndex: number) => void;
  onRegenerate: (sourceText: string) => void;
}

const DEFER_MESSAGE_COUNT = 36;
const DEFER_MESSAGE_CHARS = 20000;

function getBlockTextLength(block: MessageBlock) {
  if (block.kind === 'text' || block.kind === 'reasoning') {
    return block.text.length;
  }

  return 0;
}

function isHeavyMessage(record: MessageRecord) {
  const blockTextLength = record.blocks.reduce((total, block) => total + getBlockTextLength(block), 0);
  return record.text.length + blockTextLength > DEFER_MESSAGE_CHARS;
}

export default function MessageList({
  listRef,
  messages,
  onCopy,
  onDeleteMessage,
  onEdit,
  onRegenerate,
}: MessageListProps) {
  const shouldDeferContent = messages.length > DEFER_MESSAGE_COUNT || messages.some(isHeavyMessage);

  return (
    <div className="message-list" id="message-list" ref={listRef}>
      {messages.map((record, index) => (
        <MessageBubble
          key={`${record.role}-${index}-${record.text.slice(0, 24)}-${record.blocks.length}`}
          deferContent={shouldDeferContent}
          messageIndex={index}
          record={record}
          onCopy={onCopy}
          onDelete={onDeleteMessage}
          onEdit={onEdit}
          onRegenerate={onRegenerate}
        />
      ))}
    </div>
  );
}
