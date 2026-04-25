import type { MouseEvent, ReactNode } from 'react';

interface DropdownProps {
  align?: 'left' | 'right';
  buttonClassName?: string;
  buttonChildren: ReactNode;
  children: ReactNode;
  disabled?: boolean;
  isOpen: boolean;
  onToggle: (event: MouseEvent<HTMLButtonElement>) => void;
}

export default function Dropdown({
  align = 'left',
  buttonClassName = 'tool-btn-capsule',
  buttonChildren,
  children,
  disabled = false,
  isOpen,
  onToggle,
}: DropdownProps) {
  return (
    <div className="dropdown-container">
      <button className={buttonClassName} disabled={disabled} type="button" onClick={onToggle}>
        {buttonChildren}
      </button>
      <div
        className={`dropdown-menu ${align === 'right' ? 'right-align' : ''} ${isOpen ? 'show' : ''}`.trim()}
        onClick={(event) => event.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
