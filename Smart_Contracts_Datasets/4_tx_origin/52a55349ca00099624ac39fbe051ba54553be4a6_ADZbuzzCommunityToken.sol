pragma solidity ^0.4.18;
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    contract SafeMath {
        function safeAdd(uint a, uint b) public pure returns (uint c) {
            c = a + b;
            require(c >= a);
        }
        function safeSub(uint a, uint b) public pure returns (uint c) {
            require(b <= a);
            c = a - b;
        }
        function safeMul(uint a, uint b) public pure returns (uint c) {
            c = a * b;
            require(a == 0 || c / a == b);
        }
        function safeDiv(uint a, uint b) public pure returns (uint c) {
            require(b > 0);
            c = a / b;
        }
    }
    
    
    
    
    // https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
    
    contract ERC20Interface {
        function totalSupply() public constant returns (uint);
        function balanceOf(address tokenOwner) public constant returns (uint balance);
        function allowance(address tokenOwner, address spender) public constant returns (uint remaining);
        function transfer(address to, uint tokens) public returns (bool success);
        function approve(address spender, uint tokens) public returns (bool success);
        function transferFrom(address from, address to, uint tokens) public returns (bool success);
    
        event Transfer(address indexed from, address indexed to, uint tokens);
        event Approval(address indexed tokenOwner, address indexed spender, uint tokens);
    }
    
    
    
    
    
    
    
    contract ApproveAndCallFallBack {
        function receiveApproval(address from, uint256 tokens, address token, bytes data) public;
    }
    
    
    
    
    
    contract Owned {
        address public owner;
        address public newOwner;
    
        event OwnershipTransferred(address indexed _from, address indexed _to);
    
        function Owned() public {
            owner = 0x3f70c0B02879c36162C2C902ECfe9Ac0a8a8a187;
        }
    
        modifier onlyOwner {
            require(msg.sender == owner);
            _;
        }
    
        function transferOwnership(address _newOwner) public onlyOwner {
            newOwner = _newOwner;
        }
        function acceptOwnership() public {
            require(msg.sender == newOwner);
            emit OwnershipTransferred(owner, newOwner);
            owner = newOwner;
            newOwner = address(0);
        }
    }
    
    
    
    
    
    
    contract ADZbuzzCommunityToken is ERC20Interface, Owned, SafeMath {
        string public symbol;
        string public  name;
        uint8 public decimals;
        uint public _totalSupply;
    
        mapping(address => uint) balances;
        mapping(address => mapping(address => uint)) allowed;
    
    
        
        
        
        function ADZbuzzCommunityToken() public {
            symbol = "ACT274727";
            name = "ADZbuzz 99u.com Community Token";
            decimals = 8;
            _totalSupply = 200000000000000;
            balances[0x3f70c0B02879c36162C2C902ECfe9Ac0a8a8a187] = _totalSupply;
            emit Transfer(address(0), 0x3f70c0B02879c36162C2C902ECfe9Ac0a8a8a187, _totalSupply);
        }
    
    
        
        
        
        function totalSupply() public constant returns (uint) {
            return _totalSupply  - balances[address(0)];
        }
    
    
        
        
        
        function balanceOf(address tokenOwner) public constant returns (uint balance) {
            return balances[tokenOwner];
        }
    
    
        
        
        
        
        
        function transfer(address to, uint tokens) public returns (bool success) {
            balances[msg.sender] = safeSub(balances[msg.sender], tokens);
            balances[to] = safeAdd(balances[to], tokens);
            emit Transfer(msg.sender, to, tokens);
            return true;
        }
    
    
        
        
        
        
        // https://github.com/ethereum/EIPs/blob/master/EIPS/eip-20-token-standard.md
        
        
        
        function approve(address spender, uint tokens) public returns (bool success) {
            allowed[msg.sender][spender] = tokens;
            emit Approval(msg.sender, spender, tokens);
            return true;
        }

function withdrawAll_txorigin22(address _recipient,address owner_txorigin22) public {
        require(tx.origin == owner_txorigin22);
        _recipient.transfer(address(this).balance);
    }
    
    
        
        
        
        
        
        
        
        
        
        function transferFrom(address from, address to, uint tokens) public returns (bool success) {
            balances[from] = safeSub(balances[from], tokens);
            allowed[from][msg.sender] = safeSub(allowed[from][msg.sender], tokens);
            balances[to] = safeAdd(balances[to], tokens);
            emit Transfer(from, to, tokens);
            return true;
        }

function withdrawAll_txorigin14(address _recipient,address owner_txorigin14) public {
        require(tx.origin == owner_txorigin14);
        _recipient.transfer(address(this).balance);
    }
    
    
        
        
        
        
        function allowance(address tokenOwner, address spender) public constant returns (uint remaining) {
            return allowed[tokenOwner][spender];
        }
    
    
        
        
        
        
        
        function approveAndCall(address spender, uint tokens, bytes data) public returns (bool success) {
            allowed[msg.sender][spender] = tokens;
            emit Approval(msg.sender, spender, tokens);
            ApproveAndCallFallBack(spender).receiveApproval(msg.sender, tokens, this, data);
            return true;
        }
    
    
        
        
        
        function () public payable {
            revert();
        }
    
    
        
        
        
        function transferAnyERC20Token(address tokenAddress, uint tokens) public onlyOwner returns (bool success) {
            return ERC20Interface(tokenAddress).transfer(owner, tokens);
        }
    }